from typing import Optional
import argparse
from dataclasses import dataclass, asdict


@dataclass
class CFG:
    output_filepath: str = 'output/submission.json'
    # Model
    model_path: str = 'output/pretrained_from_scratch'
    adapter_path: Optional[str] = None
    max_model_len: int = 10240 #61000 for phi-3
    grid_encoder: str = 'GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))'
    prompt_version: str = 'output-from-examples-v0'
    # Dataset
    dataset_path: str = './ood_TTT_data1.json'
    n_tasks: Optional[int] = None # Optional parameter to limit the number of task in the inference, set it to None to use all the tasks
    # Inference params
    max_output_tokens: int = 1100 # Maximum number of tokens to generate
    predictions_per_task: int = 8 # How many predictions to make for each task, ideally should be a multiple of 8 (the number of geometric data augmentations)
    best_of: int = 1 # controls the number of beams used in beam search
    temperature: float = 0.0 # temperature for sampling, 0.0 for greedy search
    n_gpus: int = 1
    tensor_parallel_size: int = 1
    n: int = 1 # number of samples to generate
    batch_size: int = 512 # batch size for inference
    swap_space: int = 4 # CPU swap space size (GiB) per GPU. Default: 4
    random_seed: Optional[int] = None # random seed for data augmentation
    verbose: bool = False


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument('--model_path', type=str, help="Path to the model")
    parser.add_argument('--adapter_path', type=str, help="Path to the LoRA adapter")
    parser.add_argument('--max_model_len', type=int, help="Maximum number of tokens in the model")
    parser.add_argument('--grid_encoder', type=str, help="Name of the grid encoder")
    parser.add_argument('--prompt_version', type=str, help="Prompt version")
    parser.add_argument('--dataset_path', type=str, help="Path to the dataset to make inference")
    parser.add_argument('--n_tasks', type=int, help="If given only the first n tasks will be evaluated")
    parser.add_argument('--output_filepath', type=str, help="Path to the json file with the predictions")
    parser.add_argument('--predictions_per_task', type=int, help="Number of predictions per task, use a multiple of 8")
    parser.add_argument('--best_of', type=int, help="controls the number of beams used in beam search")
    parser.add_argument('--temperature', type=float, help="temperature for sampling, 0.0 for greedy search")
    parser.add_argument('--n', type=int, help="number of samples to generate")
    parser.add_argument('--batch_size', type=int, help="batch size for inference")
    parser.add_argument('--max_output_tokens', type=int, help="Maximum number of tokens to generate")
    parser.add_argument('--random_seed', type=int, help="Random seed for data augmentation")
    parser.add_argument('--swap_space', type=int, help="CPU swap space size (GiB) per GPU")
    parser.add_argument('--verbose', action='store_true', help="Print verbose output")
    return parser.parse_args()


import json
import os
import numpy as np
from tqdm.auto import tqdm
from itertools import islice, product

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer, AutoConfig

from arc24.data_augmentation import (
    get_random_color_map, set_random_seed)
from arc24.prompting import parse_grid_from_response, print_smallest_prompt, create_prompts_from_task
from arc24.encoders import create_grid_encoder
from arc24.logging import log_execution_time, logging

logger = logging.getLogger(__name__)


@log_execution_time
def inference_main():
    # Override default configuration using arguments
    args = parse_args()
    cfg = CFG(**{k: v for k, v in vars(args).items() if v is not None})
    logger.info(f'Inference configuration: {asdict(cfg)}')

    with open(cfg.dataset_path) as f:
        data = json.load(f)
    if cfg.n_tasks is not None and cfg.n_tasks > 0:
        data = dict(islice(data.items(), cfg.n_tasks))
    logger.info(f'There are {len(data)} tasks to solve in {cfg.dataset_path}')

    #tensor_parallel_size = get_tensor_parallel_size(cfg.model_path)
    #logger.info(f'Loading {cfg.model_path} with tensor_parallel_size={tensor_parallel_size}')
    llm = LLM(model=cfg.model_path,
                trust_remote_code=True,
                dtype='half', # this is causing NaNs on my computer due to using float16 instead of bfloat16
                # dtype='auto',
                #tensor_parallel_size=tensor_parallel_size, # to use 2 gpus
                max_model_len=cfg.max_model_len,
                #kv_cache_dtype='fp8_e5m2', I have disabled kv cache quantization because it is hurtful
                enforce_eager=True, # without this 13.9GB of memory is used on each GPU, with this is 13.3GB,
                disable_log_stats=True,
                max_num_seqs=255, # default is supposed to be 256 I have used it to solve some weird illegal memory error
                swap_space=cfg.swap_space, # CPU swap space size (GiB) per GPU, has great influence on RAM but I haven't noticed any performance difference
                enable_lora=cfg.adapter_path is not None,
                )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    # Fix chat template if it's not set
    if tokenizer.chat_template is None:
        logger.warning('The tokenizer does not have a chat template, assigning Qwen2 chat template')
        reference_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
        tokenizer.chat_template = reference_tokenizer.chat_template
    
    set_random_seed(cfg.random_seed)
    grid_encoder = create_grid_encoder(cfg.grid_encoder)
    prompts_conf = create_prompts(
        data, grid_encoder, tokenizer, cfg.prompt_version, cfg.predictions_per_task)
    prompts = [conf['prompt'] for conf in prompts_conf]
    if cfg.verbose: print_smallest_prompt(prompts)

    sampling_params = get_sampling_params(cfg.best_of, cfg.temperature, cfg.n, cfg.max_output_tokens)
    outputs = generate_outputs_with_batches(
        llm, prompts, sampling_params, batch_size=cfg.batch_size, adapter_path=cfg.adapter_path)
    task_results = create_tasks_results(
        outputs=outputs, prompts_conf=prompts_conf, grid_encoder=grid_encoder,
        prompt_version=cfg.prompt_version, data=data, verbose=cfg.verbose)
    solutions = create_solutions(task_results, data)

    with open(cfg.output_filepath, 'w') as f:
        json.dump(solutions, f)
    with open(cfg.output_filepath.replace('.json', '_task_results.json'), 'w') as f:
        json.dump(task_results, f)

    # Cleanup GPU memory - handle different vllm versions
    try:
        if hasattr(llm.llm_engine, 'model_executor'):
            del llm.llm_engine.model_executor
    except AttributeError:
        pass
    del llm

    clear_vllm_gpu_memory()


def get_tensor_parallel_size(model_path):
    config = AutoConfig.from_pretrained(model_path)
    if hasattr(config, 'num_attention_heads'):
        if config.num_attention_heads % 2 == 1:
            return 1
    return 2


def create_prompts(data, grid_encoder, tokenizer, prompt_version, predictions_per_task):
    prompts = []
    for task_id, task in tqdm(data.items(), total=len(data), desc='Creating prompts'):
        data_augmentation_params = list(product([False, True], [0, 1, 2, 3]))
        for hflip, n_rot90 in data_augmentation_params:
            for _ in range(predictions_per_task//len(data_augmentation_params)):
                color_map = get_random_color_map(change_background_probability=0.1)
                data_augmentation_kwargs = dict(hflip=hflip, n_rot90=n_rot90, color_map=color_map)
                task_prompts = create_prompts_from_task(
                    task, grid_encoder=grid_encoder, tokenizer=tokenizer,
                    is_train_prompt=False, prompt_version=prompt_version)
                for idx, prompt in enumerate(task_prompts):
                    prompts.append(dict(task_id=task_id,
                                        data_augmentation_kwargs=data_augmentation_kwargs,
                                        prompt=prompt,
                                        idx=idx))
    return prompts


def get_sampling_params(best_of, temperature, n, max_output_tokens):
    # # https://docs.vllm.ai/en/latest/dev/sampling_params.html
    if best_of == 1:
        sampling_params = SamplingParams(n=n, temperature=temperature, max_tokens=max_output_tokens, logprobs=0,
                                         skip_special_tokens=False)
    else:
        sampling_params = SamplingParams(n=n, temperature=0.0, max_tokens=max_output_tokens,
                              use_beam_search=True, best_of=best_of, logprobs=0)
    logger.info(f'VLLM Sampling params: {sampling_params}')
    return sampling_params


def generate_outputs_with_batches(llm, prompts, sampling_params, batch_size=512, adapter_path=None):
    if adapter_path is None:
        lora_request = None
    else:
        lora_request = LoRARequest(lora_name='lora', lora_int_id=1, lora_path=adapter_path)
    outputs = []
    logger.info(f'Generating outputs with batch_size={batch_size}, there are {len(prompts)} prompts')
    for i in tqdm(range(0, len(prompts), batch_size), desc='Generating outputs with batches', smoothing=0):
        batch = prompts[i:i+batch_size]
        if batch:
            outputs += llm.generate(batch, sampling_params, use_tqdm=True, lora_request=lora_request)
    return outputs


def create_tasks_results(outputs, prompts_conf, grid_encoder, prompt_version, data, verbose=False):
    task_results = prompts_conf.copy()
    for idx, output in tqdm(enumerate(outputs), total=len(outputs), desc='Parsing outputs'):
        task_id = prompts_conf[idx]['task_id']
        data_augmentation_kwargs = prompts_conf[idx]['data_augmentation_kwargs']
        sample_idx = prompts_conf[idx]['idx']
        response = output.outputs[0].text
        if idx == 0:
            logger.info(f'Parsing output from {task_id}_{sample_idx}: {response}')
        try:
            if prompt_version.startswith('code-from-examples'):
                # TODO: it would be more efficient to solve the whole task at once, not just one sample
                from omniarc.execution import safe_execute_predicted_code
                code = response.split('```')[0]
                predicted_task = safe_execute_predicted_code(code, data[task_id], func_name='task')
                if predicted_task['train'] == data[task_id]['train']:
                    grid = predicted_task['test'][sample_idx]['output']
                else:
                    grid = []
            else:
                grid = parse_grid_from_response(response, grid_encoder)
            validate_grid(grid)
        except Exception as e:
            # TODO: better exception printing (shape of the grid)
            if verbose: print(f'Exception when parsing response from {task_id}_{sample_idx}: {e} \n{response}')
            grid = []
        task_results[idx]['grid'] = grid
        task_results[idx]['response'] = response
        task_results[idx]['cumulative_logprob'] = output.outputs[0].cumulative_logprob
        task_results[idx]['n_tokens'] = len(output.outputs[0].token_ids)
    return task_results


def validate_grid(grid):
    assert isinstance(grid, list), f'Grid is not a list: {grid}'
    grid = np.array(grid, dtype=np.int8)
    assert grid.ndim == 2, f'Grid has more than 2 dimensions: {grid.ndim}'
    assert grid.shape[0] > 0, f'Grid has 0 rows: {grid.shape}'
    assert grid.shape[1] > 0, f'Grid has 0 columns: {grid.shape}'
    assert grid.min() >= 0, f'Grid has negative values: {grid.min()}'
    assert grid.max() < 10, f'Grid has values greater than 9: {grid.max()}'


def create_solutions(task_results, data):
    solutions = _create_empty_solutions(data)
    for task_result in task_results:
        task_id = task_result['task_id']
        sample_idx = task_result['idx']
        attempt_name = f"attempt_{len(solutions[task_id][sample_idx]) + 1}"
        solutions[task_id][sample_idx][attempt_name] = task_result['grid']
    return solutions


def _create_empty_solutions(data):
    solutions = dict()
    for task_id, task in data.items():
        solutions[task_id] = [dict() for _ in task['test']]
    return solutions


def clear_vllm_gpu_memory():
    # https://github.com/vllm-project/vllm/issues/1908
    from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
    import torch
    import gc
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    inference_main()