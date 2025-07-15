
import os
import random
import json
import glob
import numpy as np
from tqdm.auto import tqdm
import wandb
from typing import Optional, List
import argparse
import traceback
from functools import partial
from dataclasses import dataclass, asdict, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datasets import Dataset, IterableDataset

from arc24.encoders import create_grid_encoder
from arc24.data_augmentation import (
    random_augment_task_star,
    set_random_seed,
    random_compose_new_task_by_adding_additional_transformation
)
from arc24.prompting import create_prompts_from_task, print_smallest_prompt, pretty_print_prompt
from arc24.data import load_arc_data_with_solutions, BarcDataset
from arc24.logging import log_execution_time, logging

from accelerate.logging import get_logger
from accelerate import Accelerator

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']
    
logger = get_logger(__name__)
# logger = logging.getLogger(__name__)

# grid encoder experiments
@dataclass
class CFG:
    verbose: bool = True
    resume_from_checkpoint: bool = True
    model_path: str = 'output/pretrained_from_scratch'
    adapter_path: Optional[str] = None
    use_4bit_quantization: bool = False
    train_datasets: List[List[str]] = field(default_factory=lambda: [['ood_TTT_data1-00000000-train.json', 'output-from-examples-v0']])
    remove_train_samples_to_fit_max_seq_len: bool = False
    subsample_train_tasks_ratio: Optional[float] = None
    val_dataset: List[str] = field(default_factory=lambda: ['ood_TTT_data1-00000000-val.json', 'output-from-examples-v0'])
    output_dir: str = './output/'
    n_gpus: int = 1
    device_map: str = 'custom' # 'custom', 'balanced', 'auto', 'None'
    max_seq_len: int = 4096
    epochs = 0
    max_steps : Optional[int] =  100
    logging_steps: int = 1 #10a
    eval_steps: int = 10 #50
    save_steps: Optional[int] = None
    report_to: str = 'wandb'
    warmup_ratio: float = 0.05
    batch_size: int = 16 #16
    random_seed: Optional[int] = None # None, 7
    grid_encoder: str = 'GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))'
    # SmolLM-135M-Instruct: (4, 4); Qwen/Qwen2-0.5B-Instruct: (1, 2)
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size = 1 # if using 2 the validation loss is not correctly computed
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "linear" #linear, constant_with_warmup, cosine, cosine_with_restarts
    lr_num_cycles: int = 4 # only applicable for cosine_with_restarts
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit" # "paged_adamw_8bit"
    torch_dtype: str = "bfloat16" # "bfloat16" or "float16", float16 causes divergence when training on my PC, but it is 4x faster on Kaggle
    # LoRA
    use_lora: bool = True
    use_rslora = True,
    use_dora = True,
    lora_r: int = 32
    lora_weight_initialization: str = 'default' # 'gaussian', 'olora', 'pissa', 'pissa_niter_[number of iters]', 'loftq', 'default'
    # Data augmentation
    compose_new_task_probability: float = 0.0
    compose_new_task_weights: Optional[List[float]] = None
    # Verify
    verify_correct_output_probability: float = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument('--model_path', type=str, help="Path to the model")
    parser.add_argument('--adapter_path', type=str, help="Path to the LoRA adapter for initialization")
    parser.add_argument('--use_4bit_quantization', action='store_true', help="Whether to use 4-bit quantization")
    parser.add_argument('--output_dir', type=str, help="Path to the output LoRA")
    parser.add_argument('--train_datasets', nargs='+', action='append',
                        metavar=('filepath', 'prompt_version'), help="Path to the datasets for training")
    parser.add_argument('--val_dataset', type=str, nargs=2,
                        metavar=('filepath', 'prompt_version'), help="Path to the dataset for validation")
    parser.add_argument('--max_steps', type=int, help="Max steps to fine-tune")
    parser.add_argument('--warmup_ratio', type=float, help="Warmup ratio, relative to training steps")
    parser.add_argument('--max_seq_len', type=int, help="Max sequence length in tokens")
    parser.add_argument('--eval_steps', type=int, help="Number of steps between evaluations")
    parser.add_argument('--save_steps', type=int, help="Number of steps between saving checkpoints")
    parser.add_argument('--logging_steps', type=int, help="Number of steps between logging")
    parser.add_argument('--learning_rate', type=float, help='Learning rate for fine-tuning')
    parser.add_argument('--lr_scheduler_type', type=str, help='Learning rate scheduler type')
    parser.add_argument('--lr_num_cycles', type=int, help='Number of cycles for cosine_with_restarts scheduler')
    parser.add_argument('--gradient_checkpointing', action=argparse.BooleanOptionalAction, help='Whether to use gradient checkpointing')
    parser.add_argument('--batch_size', type=int, help='Batch size for fine-tuning')
    parser.add_argument('--per_device_train_batch_size', type=int, help='Batch size per device for fine-tuning')
    parser.add_argument('--report_to', type=str, help="Set it to tensorboard to disable wandb")
    parser.add_argument('--torch_dtype', type=str, help="Which dtype to use with torch")
    parser.add_argument('--lora_r', type=int, help="Rank of the LoRA adapter")
    parser.add_argument('--lora_weight_initialization', type=str, help="Weight initialization for LoRA")
    parser.add_argument('--n_gpus', type=int, help="Number of gpus to use")
    parser.add_argument('--device_map', type=str, help="Device map for the model, could be 'balanced', 'auto', 'custom', or 'None'")
    parser.add_argument('--grid_encoder', type=str, help="Name of the grid encoder")
    parser.add_argument('--remove_train_samples_to_fit_max_seq_len', action='store_true',
                        help="Whether to remove training samples to fit max_seq_len")
    parser.add_argument('--subsample_train_tasks_ratio', type=float, help="Ratio of train tasks to subsample, 1 means no subsampling")
    parser.add_argument('--random_seed', type=int, help="Random seed for data generation")
    parser.add_argument('--resume_from_checkpoint', action=argparse.BooleanOptionalAction, help="Whether to resume from checkpoint")
    parser.add_argument('--compose_new_task_probability', type=float, help="Probability of composing a new task")
    parser.add_argument('--compose_new_task_weights', nargs='+', type=float, help="Weights for composing a new task")
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, help="Whether to print verbose information")
    parser.add_argument('--use_lora', action=argparse.BooleanOptionalAction, help="Whether to use LoRA")
    parser.add_argument('--verify_correct_output_probability', type=float, help="Probability of using a correct output when verifying")
    parser.add_argument('--optim', type=str, help="Optimizer to use")
    return parser.parse_args()


@log_execution_time
def fine_tuning_main():
    # Override default configuration using arguments
    cfg = CFG(**{k: v for k, v in vars(parse_args()).items() if v is not None})
    save_train_conf(cfg)
    if cfg.report_to == 'wandb':
        accelerator = Accelerator(log_with=cfg.report_to)
        accelerator.init_trackers(
            project_name=os.path.basename(os.path.dirname(cfg.output_dir)),
            config=cfg,
            init_kwargs={"wandb": dict(dir=cfg.output_dir,
                                       name=os.path.basename(cfg.output_dir))}
        )
    else:
        accelerator = Accelerator()
    logger.info(f'Train configuration: {asdict(cfg)}')

    model = get_model(cfg.model_path, n_gpus=cfg.n_gpus, torch_dtype=cfg.torch_dtype,
                      use_4bit_quantization=cfg.use_4bit_quantization, device_map=cfg.device_map)
    tokenizer = get_tokenizer(cfg.model_path, model)
    if cfg.use_lora:
        model = get_lora_model(model, cfg.adapter_path, cfg.lora_r, cfg.use_rslora,
                               cfg.use_dora, cfg.lora_weight_initialization)
    else:
        logger.info('Not using LoRA, full model will be fine-tuned')

    grid_encoder = create_grid_encoder(cfg.grid_encoder)
    dataset_kwargs = {'grid_encoder': grid_encoder, 'tokenizer': tokenizer, 'max_seq_len': cfg.max_seq_len, 'verbose': cfg.verbose}
    train_dataset = IterableDataset.from_generator(
        # for some weird reason, it does not work correctly with lists and I have to use partial with the lists
        partial(random_prompt_generator, train_datasets=cfg.train_datasets,
                compose_new_task_weights=cfg.compose_new_task_weights,
                random_seed=cfg.random_seed, **dataset_kwargs,
                remove_train_samples_to_fit_max_seq_len=cfg.remove_train_samples_to_fit_max_seq_len,
                subsample_tasks_ratio=cfg.subsample_train_tasks_ratio,
                compose_new_task_probability=cfg.compose_new_task_probability,
                verify_correct_output_probability=cfg.verify_correct_output_probability))
    val_dataset = create_validation_dataset(*cfg.val_dataset, **dataset_kwargs)

    training_arguments = get_training_arguments(cfg)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=get_data_collator(tokenizer),
        args=training_arguments,
    )
    if cfg.lr_scheduler_type == 'cyclic':
        replace_trainer_lr_scheduler_with_cyclic_lr(
            trainer, cfg.warmup_ratio, cfg.learning_rate, cfg.lr_num_cycles)
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint and is_checkpoint_available(cfg.output_dir))



# Model
def get_device_map(n_gpus, model_path, device_map):
    if n_gpus == 2 and device_map == 'custom':
        if 'llama' in model_path.lower():
            logger.info('Using llama custom device map')
            device_map = {
                'model.embed_tokens': 0,
                'model.layers.0': 0,
                'model.layers.1': 0,
                'model.layers.2': 0,
                'model.layers.3': 0,
                'model.layers.4': 0,
                'model.layers.5': 0,
                'model.layers.6': 0,
                'model.layers.7': 0,
                'model.layers.8': 0,
                'model.layers.9': 0,
                'model.layers.10': 0,
                'model.layers.11': 0,
                'model.layers.12': 0,
                'model.layers.13': 0,
                'model.layers.14': 0,
                'model.layers.15': 0,
                'model.layers.16': 0,
                'model.layers.17': 1,
                'model.layers.18': 1,
                'model.layers.19': 1,
                'model.layers.20': 1,
                'model.layers.21': 1,
                'model.layers.22': 1,
                'model.layers.23': 1,
                'model.layers.24': 1,
                'model.layers.25': 1,
                'model.layers.26': 1,
                'model.layers.27': 1,
                'model.layers.28': 1,
                'model.layers.29': 1,
                'model.layers.30': 1,
                'model.layers.31': 1,
                'model.norm': 1,
                'model.rotary_emb': 1,
                'lm_head': 1,
            }
        elif 'qwen2' in model_path.lower() and '0.5b' in model_path.lower():
            logger.info('Using qwen2-0.5b-instruct custom device map')
            device_map = {
                'model.embed_tokens': 0,
                'lm_head': 0,
                'model.layers.0': 0,
                'model.layers.1': 0,
                'model.layers.2': 0,
                'model.layers.3': 0,
                'model.layers.4': 0,
                'model.layers.5': 0,
                'model.layers.6': 0,
                'model.layers.7': 0,
                'model.layers.8': 1,
                'model.layers.9': 1,
                'model.layers.10': 1,
                'model.layers.11': 1,
                'model.layers.12': 1,
                'model.layers.13': 1,
                'model.layers.14': 1,
                'model.layers.15': 1,
                'model.layers.16': 1,
                'model.layers.17': 1,
                'model.layers.18': 1,
                'model.layers.19': 1,
                'model.layers.20': 1,
                'model.layers.21': 1,
                'model.layers.22': 1,
                'model.layers.23': 1,
                'model.norm': 1,
                'model.rotary_emb': 1,
            }
        elif 'qwen2' in model_path.lower() and '1.5b' in model_path.lower():
            logger.info('Using qwen2-1.5b-instruct custom device map')
            device_map = {
                'model.embed_tokens': 0,
                'lm_head': 0,
                'model.layers.0': 0,
                'model.layers.1': 0,
                'model.layers.2': 0,
                'model.layers.3': 0,
                'model.layers.4': 0,
                'model.layers.5': 0,
                'model.layers.6': 0,
                'model.layers.7': 0,
                'model.layers.8': 0,
                'model.layers.9': 0,
                'model.layers.10': 0,
                'model.layers.11': 1,
                'model.layers.12': 1,
                'model.layers.13': 1,
                'model.layers.14': 1,
                'model.layers.15': 1,
                'model.layers.16': 1,
                'model.layers.17': 1,
                'model.layers.18': 1,
                'model.layers.19': 1,
                'model.layers.20': 1,
                'model.layers.21': 1,
                'model.layers.22': 1,
                'model.layers.23': 1,
                'model.layers.24': 1,
                'model.layers.25': 1,
                'model.layers.26': 1,
                'model.layers.27': 1,
                'model.norm': 1,
                'model.rotary_emb': 1,
                }
        elif 'qwen2' in model_path.lower() and '7b' in model_path.lower():
            logger.info('Using qwen2-7b-instruct custom device map')
            device_map = {
                'model.embed_tokens': 0,
                'model.layers.0': 0,
                'model.layers.1': 0,
                'model.layers.2': 0,
                'model.layers.3': 0,
                'model.layers.4': 0,
                'model.layers.5': 0,
                'model.layers.6': 0,
                'model.layers.7': 0,
                'model.layers.8': 0,
                'model.layers.9': 0,
                'model.layers.10': 0,
                'model.layers.11': 0,
                'model.layers.12': 0,
                'model.layers.13': 0,
                'model.layers.14': 1,
                'model.layers.15': 1,
                'model.layers.16': 1,
                'model.layers.17': 1,
                'model.layers.18': 1,
                'model.layers.19': 1,
                'model.layers.20': 1,
                'model.layers.21': 1,
                'model.layers.22': 1,
                'model.layers.23': 1,
                'model.layers.24': 1,
                'model.layers.25': 1,
                'model.layers.26': 1,
                'model.layers.27': 1,
                'model.norm': 1,
                'model.rotary_emb': 1,
                'lm_head': 1}
        elif 'nano' in model_path.lower() or 'smollm' in model_path.lower():
            # quick fix for Kaggle
            logger.warning(f'Custom device map was asked, but instead balanced was returned as quick fix: {model_path}')
            device_map = 'balanced'
        else:
            raise NotImplementedError(f'Custom device map not implemented for {model_path}')
    else:
        if device_map == 'None':
            logger.info('Using None device map')
            device_map = None
        elif device_map in ['balanced', 'auto']:
            logger.info(f'Using {device_map} device map')
        elif device_map == 'custom':
            logger.warning('Custom device map is not implemented for n_gpus != 2, using auto device map instead')
            device_map = 'auto'
        else:
            raise ValueError(f'Unknown device map {device_map}')
    return device_map


def get_flash_attention_implementation():
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
    except ImportError:
        attn_implementation = None
    logger.info(f'Using {attn_implementation} attention implementation')
    return attn_implementation


def get_torch_dtype(torch_dtype):
    if torch_dtype == 'float16':
        logger.info('Using float16 torch dtype')
        return torch.float16
    elif torch_dtype == 'bfloat16':
        logger.info('Using bfloat16 torch dtype')
        return torch.bfloat16
    else:
        raise ValueError(f'Unknown torch dtype {torch_dtype}')


def get_model(model_path, n_gpus, torch_dtype, device_map, use_4bit_quantization=False):
    if use_4bit_quantization:
        logger.info('Using 4-bit quantization')
        bnb_config = BitsAndBytesConfig(
            load_in_4bit= True,
            bnb_4bit_quant_type= "nf4",
            bnb_4bit_compute_dtype= torch.float16,
            bnb_4bit_use_double_quant= True,
            llm_int8_enable_fp32_cpu_offload= True,
            llm_int8_skip_modules=['gate', 'lm_head'],
        )
    else:
        bnb_config = None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=get_device_map(n_gpus, model_path, device_map),
        # max_memory={0: '9GB', 1: '8GB'},
        trust_remote_code=True,
        torch_dtype=get_torch_dtype(torch_dtype), #bfloat16 is 4 times slower on Kaggle than float16, on my computer they are the same speed
        attn_implementation=get_flash_attention_implementation(),
        )
    # print(model.hf_device_map)
    print_gpu_memory()
    if use_4bit_quantization:
        # QLoRA on Kaggle is 4 times slower than LoRA, I'm trying to disable gradient checkpointing
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    return model


def get_tokenizer(model_path, model, pad_token='<|pad|>'):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True)
    if tokenizer.pad_token == tokenizer.eos_token:
        if 'qwen' in model_path.lower():
            logger.info('Changing eos token to <|im_end|> for Qwen models, because it is the same as padding token <|endoftext|>')
            tokenizer.eos_token = '<|im_end|>'
        elif 'smollm' in model_path.lower():
            logger.info('Changing pad token to "<|endoftext|>" for SmolLM models, because it is the same as eos token <|im_end|>')
            tokenizer.pad_token = "<|endoftext|>"
        else:
            raise NotImplementedError('Changing padding token is only implemented for Qwen models')
    elif 'pad_token' not in tokenizer.special_tokens_map or tokenizer.pad_token == tokenizer.eos_token:
        logger.info('Adding padding token because the tokenizer does not have one')
        assert pad_token not in tokenizer.get_vocab()
        tokenizer.add_special_tokens({'pad_token': pad_token})
        tokenizer.padding_side = 'right'
        model.resize_token_embeddings(len(tokenizer))
    # if tokenizer.chat_template is None:
    #     logger.warning('The tokenizer does not have a chat template, assigning Qwen2 chat template')
    #     tokenizer.chat_template = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True).chat_template
    #     # tried adding this additional code without success
    #     tokenizer.add_special_tokens({'eos_token': '<|im_end|>'})
    #     tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    #     # tokenizer.eos_token = '<|im_end|>'
    #     # tokenizer.pad_token = '<|endoftext|>'
    assert tokenizer.pad_token != tokenizer.eos_token
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    return tokenizer


def get_lora_model(model, adapter_path, r, use_rslora, use_dora, weight_initalization):
    if adapter_path is None:
        if weight_initalization == 'default': weight_initalization = True
        peft_config = LoraConfig(
            # lora_alpha: LoRA scaling factor.
            lora_alpha=64, #64,
            lora_dropout=0.1, # 0.1, althought Vaca suggested to use 0.05 for big models
            # r: the rank of the update matrices, expressed in int. Lower rank results in smaller update matrices with fewer trainable parameters.
            r=r, #16
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules: The modules (for example, attention blocks) to apply the LoRA update matrices.
            target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj'],
            use_rslora=use_rslora,
            use_dora=use_dora,
            init_lora_weights=weight_initalization # bool | Literal['gaussian', 'olora', 'pissa', 'pissa_niter_[number of iters]', 'loftq'] = True,
        )
        logger.info(f'Creating LoRA with the following config: {peft_config}')
        model = get_peft_model(model, peft_config)
    else:
        logger.info(f'Loading adapter from {adapter_path}')
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    return model


def print_gpu_memory():
    for device in range(torch.cuda.device_count()):
        logger.info(f'GPU {device} memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.1f} GB, max memory allocated: {torch.cuda.max_memory_allocated(device)/1024**3:.1f} GB')

# Data
def create_validation_dataset(filepath, prompt_version, grid_encoder, tokenizer, max_seq_len, verbose=False):
    if filepath.startswith('omni-arc'):
        from omniarc.dataset import create_dataset as create_omniarc_dataset
        n_tasks = int(filepath.split('-')[-1])
        logger.info(f'Creating omni-arc dataset with {n_tasks} tasks')
        omniarc_dataset = create_omniarc_dataset()
        tasks = [omniarc_dataset.sample()[1] for _ in range(n_tasks)]
    else:
        data = load_arc_data_with_solutions(filepath)
        tasks = list(data.values())
    prompts = []
    for task in tqdm(tasks, desc='create prompts'):
        prompts.extend(create_prompts_from_task(task, grid_encoder, tokenizer, prompt_version=prompt_version))
    if verbose: print_smallest_prompt(prompts)
    prompt_lengths = [len(tokenizer.encode(prompt)) for prompt in tqdm(prompts, desc='Calculating prompt lengths')]
    if verbose: print_prompt_length_percentiles(prompt_lengths, prefix='Validation')
    prompts = [prompt for prompt, prompt_length in zip(prompts, prompt_lengths) if prompt_length < max_seq_len]
    logging.info(f'Leaving {len(prompts)} validation prompts after removing those longer than {max_seq_len} tokens')
    dataset = Dataset.from_dict({'text': prompts})
    return dataset


def random_prompt_generator(train_datasets, grid_encoder, tokenizer, max_seq_len, random_seed,
                            remove_train_samples_to_fit_max_seq_len,
                            log_prompt_length_every=1000,
                            subsample_tasks_ratio=None,
                            compose_new_task_probability=0.5,
                            compose_new_task_weights=None,
                            verbose=False,
                            max_consecutive_exceptions=20,
                            verify_correct_output_probability=0.5):
    """
    """
    data = dict()
    for idx, (filepath, prompt_version) in tqdm(enumerate(train_datasets), desc='Loading training datasets'):
        if filepath.startswith('omni-arc'):
            n_tasks = int(filepath.split('-')[-1])
            from omniarc.dataset import create_dataset as create_omniarc_dataset
            omniarc_dataset = create_omniarc_dataset()
            logger.info(f'Creating omni-arc dataset with {n_tasks} tasks (real number of tasks is {len(omniarc_dataset)})')
            dataset = {f'omni-arc-{idx}|{key}|{prompt_version}': omniarc_dataset for key in range(n_tasks)}
        elif filepath.startswith('barc'):
            # example: barc-400-5-filepath
            n_tasks = int(filepath.split('-')[1])
            max_samples_per_task = int(filepath.split('-')[2])
            filepath = '-'.join(filepath.split('-')[3:])
            dataset = BarcDataset(filepath, max_samples_per_task=max_samples_per_task)
            dataset = {f'barc-{idx}|{key}|{prompt_version}': dataset for key in range(n_tasks)}
        else:
            dataset = load_arc_data_with_solutions(filepath)
            dataset = {f'{idx}|{key}|{prompt_version}': value for key, value in dataset.items()}
        logger.info(f'Loaded {len(dataset)} tasks from {filepath}')
        data.update(dataset)
    task_ids = list(data.keys())
    prompt_lengths = []
    set_random_seed(random_seed)
    if subsample_tasks_ratio is not None:
        task_ids = random.sample(task_ids, int(subsample_tasks_ratio*len(task_ids)))
        logger.info(f'Subsampled {len(task_ids)} training tasks out of a total of {len(data)}')
    else:
        logger.info(f'Using all {len(task_ids)} training tasks')
    consecutive_exceptions = 0
    sample_idx = 0
    while True:
        try:
            if len(prompt_lengths) >= log_prompt_length_every:
                temp_prompt_lengths = prompt_lengths.copy()
                prompt_lengths = []
                if verbose: print_prompt_length_percentiles(temp_prompt_lengths, prefix='Training')
                check_ratio_of_prompts_above_max_seq_len(temp_prompt_lengths, max_seq_len)
            random.shuffle(task_ids)
            for task_id in task_ids:
                sample_idx += 1
                prompt_version = task_id.split('|')[-1]
                if task_id.startswith('omni-arc'):
                    task = data[task_id].sample()[1]
                elif task_id.startswith('barc'):
                    task = data[task_id].sample()
                    #task = random_augment_task(task)
                elif prompt_version.startswith('select-output-from-examples') or prompt_version.startswith('verify-output-from-examples'):
                    task = create_random_task_for_selection_prompt(data[task_id])
                    #task = random_augment_task(task, swap_train_and_test=False)
                    # if random.random() < compose_new_task_probability:
                    #     task = random_compose_new_task_by_adding_additional_transformation(
                    #         task, weights=compose_new_task_weights, is_wrong_prediction=True)
                    if prompt_version.startswith('verify-output-from-examples'):
                        task = add_verify_output_label(task, verify_correct_output_probability=verify_correct_output_probability)
                    else:
                        task = add_correct_selection_label(task)
                else:
                    task = data[task_id]
                    if isinstance(task, list): # some datasets such as neoeye's tama have different variations of the same task
                        task = random.choice(task)
                    #if 'test' not in task and 'n_train' in task: # other datasets do not have a defined test set
                        #task = create_random_task_from_task_without_test(task)
                    task = random_augment_task_star(task)
                    #if random.random() < compose_new_task_probability:
                        #task = random_compose_new_task_by_adding_additional_transformation(
                        #    task, weights=compose_new_task_weights)
                if remove_train_samples_to_fit_max_seq_len:
                    while len(task['train']):
                        prompt, prompt_length = _create_prompt_smaller_than_max_seq_len(
                            task, grid_encoder, tokenizer, max_seq_len, prompt_version=prompt_version)
                        if prompt is not None:
                            break
                        task = remove_last_train_sample(task)
                else:
                    prompt, prompt_length = _create_prompt_smaller_than_max_seq_len(
                        task, grid_encoder, tokenizer, max_seq_len, prompt_version=prompt_version)
                prompt_lengths.append(prompt_length)
                if prompt is not None:
                    consecutive_exceptions = 0
                    if verbose and sample_idx == 1:
                        logger.info(f'Printing the first training prompt: {task_id}')
                        pretty_print_prompt(prompt)
                    yield {'text': prompt}
                else:
                    logger.debug(f'Prompt was {prompt_length}>{max_seq_len} tokens for task {task_id}, skipping task')
        except NotAvailableDataForTraining as e:
            logger.error('There is no available data for training, stopping the generator')
            raise e
        except Exception as e:
            consecutive_exceptions += 1
            logger.error(f"An error occurred when generating sample {sample_idx} (consecutive exception {consecutive_exceptions}/{max_consecutive_exceptions}): {e}")
            traceback.print_exc()
            if consecutive_exceptions >= max_consecutive_exceptions:
                raise Exception(f"{max_consecutive_exceptions} consecutive exceptions occurred.") from e


def create_random_task_from_task_without_test(task):
    """ This is useful to generate nearly infinite tasks from datasets such as RE-ARC """
    samples = np.random.choice(task['train'], task['n_train'] + 1, replace=False).tolist()
    new_task = dict(train=samples[:-1], test=samples[-1:])
    return new_task


def create_random_task_for_selection_prompt(task):
    test_candidates = [idx for idx, sample in enumerate(task['train']) if any(key.startswith('attempt_') for key in sample)]
    test_idx = random.choice(test_candidates)
    train_indices = random.sample([idx for idx in range(len(task['train'])) if idx != test_idx], k=task['n_train'])
    keep_keys = ['input', 'output']
    train_samples = []
    for idx in train_indices:
        train_samples.append({key: task['train'][idx][key] for key in keep_keys})
    test_sample = {key: task['train'][test_idx][key] for key in keep_keys}
    chosen_wrong_pred = random.choice([key for key in task['train'][test_idx] if key.startswith('attempt_')])
    test_sample['wrong_prediction'] = task['train'][test_idx][chosen_wrong_pred]
    new_task = dict(train=train_samples, test=[test_sample])
    return new_task


def add_correct_selection_label(task):
    """ Adds the fields test_correct_choice_index and test_output_choices to the task """
    if random.random() < 0.5:
        task['test_correct_choice_index'] = 1
        task['test_output_choices'] = [task['test'][0]['output'], task['test'][0]['wrong_prediction']]
    else:
        task['test_correct_choice_index'] = 2
        task['test_output_choices'] = [task['test'][0]['wrong_prediction'], task['test'][0]['output']]
    return task


def add_verify_output_label(task, verify_correct_output_probability=0.5):
    if random.random() < verify_correct_output_probability:
        task['is_test_output_correct'] = 'yes'
    else:
        task['test'][0]['output'] = task['test'][0]['wrong_prediction']
        task['is_test_output_correct'] = 'no'
    return task


def _create_prompt_smaller_than_max_seq_len(task, grid_encoder, tokenizer, max_seq_len, prompt_version):
    prompts = create_prompts_from_task(task, grid_encoder, tokenizer, prompt_version=prompt_version)
    # TODO: is this the better way to deal with multi-output tasks?
    # Should I give more weight to tasks with multiple outputs?
    prompt = random.choice(prompts)
    prompt_length = len(tokenizer.encode(prompt))
    if prompt_length < max_seq_len:
        return prompt, prompt_length
    else:
        return None, prompt_length


def remove_last_train_sample(task):
    new_task = task.copy()
    new_task['train'] = new_task['train'][:-1]
    return new_task


def print_prompt_length_percentiles(prompt_lengths, prefix):
    print(f'\t{prefix} prompt length percentiles, number of prompts: {len(prompt_lengths)}')
    for percentile in [50, 75, 90, 95, 97]:
        print(f'{prefix} prompt length percentile {percentile}: {int(np.percentile(prompt_lengths, percentile))}')
    print(f'{prefix} prompt length max: {max(prompt_lengths)}')


class NotAvailableDataForTraining(Exception):
    pass

def check_ratio_of_prompts_above_max_seq_len(prompt_lengths, max_seq_len, max_allowed_ratio=0.9):
    ratio = np.mean(np.array(prompt_lengths) > max_seq_len)
    logger.info(f'Ratio of prompts above max_seq_len: {ratio:.1%}')
    if ratio > max_allowed_ratio:
        raise NotAvailableDataForTraining(f'Too many prompts above max_seq_len: {ratio:.1%}')

# Train
def get_data_collator(tokenizer):
    if '<|start_header_id|>' in tokenizer.chat_template and '<|end_header_id|>' in tokenizer.chat_template:
        logger.info('Using llama template for collator')
        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            instruction_template='<|start_header_id|>user<|end_header_id|>',
            response_template='<|start_header_id|>assistant<|end_header_id|>',
        )
    elif '<|im_start|>' in tokenizer.chat_template:
        logger.info('Using SmolLM\Qwen template for collator')
        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            instruction_template='<|im_start|>user',
            response_template='<|im_start|>assistant',
        )
    elif '<|user|>' in tokenizer.chat_template and '<|assistant|>' in tokenizer.chat_template:
        logger.info('Using Phi-3 template for collator')
        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            instruction_template='<|user|>',
            response_template='<|assistant|>'
        )
    else:
        raise NotImplementedError(f'Tokenizer chat template not recognized: {tokenizer.chat_template}')
    return data_collator


def get_training_arguments(cfg):
    gradient_accumulation_steps = get_gradient_accumulation_steps(
        cfg.batch_size, cfg.per_device_train_batch_size, cfg.n_gpus, cfg.device_map)
    batch_size_kwargs = dict(
        # 4-16 batch size should be fine for lora.
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
    )
    scheduler_type = cfg.lr_scheduler_type
    if scheduler_type == 'cyclic':
        logger.info('Using cyclic learning rate scheduler (renaming to linear because it will be hacked later)')
        scheduler_type = 'linear'

    lr_scheduler_kwargs = {}
    if cfg.lr_scheduler_type == 'cosine_with_restarts':
        lr_scheduler_kwargs['num_cycles'] = cfg.lr_num_cycles
    training_arguments = SFTConfig(
            output_dir=cfg.output_dir,
            save_total_limit=3, # I'm only interested in the last checkpoint, I will be saving 3 to avoid corruption problems (2 will be enough for this)
            num_train_epochs=cfg.epochs,
            max_steps=cfg.max_steps,
            warmup_ratio=cfg.warmup_ratio,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=scheduler_type, #constant_with_warmup, cosine, cosine_with_restarts
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            gradient_checkpointing=cfg.gradient_checkpointing,
            optim=cfg.optim,
            max_grad_norm=cfg.max_grad_norm,

            dataset_text_field="text",
            max_seq_length=cfg.max_seq_len,

            do_eval=True,
            eval_strategy="steps",
            save_steps=cfg.save_steps or cfg.eval_steps,
            logging_steps=cfg.logging_steps, #50,
            eval_steps=cfg.eval_steps,
            log_level="info",
            report_to=cfg.report_to,

            # parameters added to make the code work with accelerate
            #dispatch_batches=False,
            # https://huggingface.co/transformers/v4.9.1/main_classes/trainer.html#trainingarguments
            ddp_find_unused_parameters=False, # only used with accelerate, got a warning saying that it slows down if True

            ignore_data_skip=True, # otherwise it takes too long to start training when resuming from checkpoint

            **batch_size_kwargs
    )
    return training_arguments


def get_gradient_accumulation_steps(batch_size, per_device_train_batch_size, n_gpus, device_map):
    if n_gpus > 1 and device_map == 'None': # multi-gpu accelerate training
        accumulation_steps = batch_size//per_device_train_batch_size//n_gpus
    else:
        accumulation_steps = batch_size//per_device_train_batch_size
    logger.info(f'Using {accumulation_steps} gradient accumulation steps')
    return accumulation_steps



def save_train_conf(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, 'cfg.json'), 'w') as f:
        json.dump({key:value for key, value in cfg.__dict__.items() if not key.startswith('__')}, f, indent=4)


def replace_trainer_lr_scheduler_with_cyclic_lr(trainer, warmup_ratio, learning_rate, num_cycles):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        cycle_steps = num_training_steps//num_cycles
        step_size_up = int(cycle_steps*warmup_ratio)
        step_size_down = cycle_steps - step_size_up
        self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=learning_rate//100,  # minimum learning rate
            max_lr=learning_rate,        # maximum learning rate
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            scale_fn=lambda x: 1.0 - (x - 1.)/num_cycles,
            scale_mode='cycle',
        )
        return self.lr_scheduler
    trainer.create_scheduler = create_scheduler.__get__(trainer)


def is_checkpoint_available(output_dir):
    is_checkpoint_available = len(glob.glob(os.path.join(output_dir, 'checkpoint-*'))) > 0
    if is_checkpoint_available:
        logger.info('Checkpoint found, resuming training')
    else:
        logger.info('No checkpoint found, starting training from scratch')
    return is_checkpoint_available


if __name__ == '__main__':
    fine_tuning_main()
