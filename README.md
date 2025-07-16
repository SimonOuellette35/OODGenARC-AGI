## GridCoder 2 results

These experiments were run on A40 RunPod instances.

### Setup
1. clone this repo
2. git clone https://github.com/SimonOuellette35/ARC_gym (tested on commit #f135ae35e01a27aaf4cec0bea4250d27c318e9f7)
3. cd ARC_gym
4. pip install -e .
5. cd ../OODGenARC-AGI
6. git clone https://github.com/SimonOuellette35/AmotizedDSL.git (tested on commit #4a13a60183d69d2318088ff456bfa584245a8830)
7. cd AmotizedDSL
8. pip install -e .
9. cd ..
10. git clone https://github.com/arcprize/ARC-AGI-2.git
11. pip install -r requirements.txt
    
The resulting folder structure should be:

    workspace/
    |
    |-- ARC_gym/
    |-- OODGenARC-AGI/
             |
             |-- AmotizedDSL/
             |-- ARC-AGI-2/

### Model checkpoint
You can train the model by following these instructions:
1. Generate training data by calling: python generate_toy_data.py (this will create training.json and validation.json)
2. Train the model by calling: python3 train_gridcoder.py (this will output gridcoder2.pth)

Or you can download the pre-trained checkpoint for these experiments: https://drive.google.com/file/d/18-9MeY9mw4JyZ1wVBJkHyMwyGBgDamvn/view?usp=sharing

You should name the checkpoint file: gridcoder2.pth

### Generating the results
 
    python test_gridcoder.py --dataset ood_data1.json
    python test_gridcoder.py --dataset ood_data2.json
    python test_gridcoder.py --dataset ood_data3.json
    python test_gridcoder.py --dataset ood_data4.json
    python test_gridcoder.py --dataset ood_data5.json
    python test_gridcoder.py --dataset ood_data6.json
    python test_gridcoder.py --dataset ood_data7.json
   
Each live above gives the success rate for each OOD Task from 1 to 7 inclusively.

## TTFT results

### Initial setup
1. Clone the following repo: https://github.com/ironbar/arc24
2. Generate the train/val/test split for the OOD tasks in the following way: go to OODGenARC-AGI repo folder, and do:
```
     python prep_fine-tuning.py ood_TTT_data1.json
```
3. Repeat the above for the 6 other task files: ood_TTT_data2/3/4/5/6/7.json.
4. Move the OOD_TTT_data*.json files to the arc24/scripts folder
5. From OODGenARC-AGI repo, run: python generate_TTT_data.py (to generate fine-tuning data for the TTFT model)
6. Move the generated training_TTT.json and validation_TTT.json files over to arc24/scripts
7. Under arc24/ run pip install -r requirements.txt
8. Some issues I faced:
   
   a) had to install: pip install torch==2.5.0 transformers==4.52.1
   
   b) had to install: pip install vllm
   
   c) when running inference.py I had a weird exception that I resolved by doing: pip install --upgrade vllm

   d) fine-tuning.py: had to remove the dispatch_batches argument at line 738

   e) inference.py: had to comment out tensor_parallel_size=tensor_parallel_size on line 91 since I'm using only 1 GPU

   f) inference.py: had to add:
```
    if tokenizer.chat_template is None:
        logger.warning('The tokenizer does not have a chat template, assigning Qwen2 chat template')
        reference_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
        tokenizer.chat_template = reference_tokenizer.chat_template

    after tokenizer = AutoTokenizer.from_pretrained(cfg.model_path) (line 101)

```

   g) inference.py: had to add:
    
```
    # Cleanup GPU memory - handle different vllm versions
    try:
        if hasattr(llm.llm_engine, 'model_executor'):
            del llm.llm_engine.model_executor
    except AttributeError:
        pass

    instead of del llm.llm_engine.model_executor around line 128
```


### LLM+TTFT results
Instead of going through steps 1 and 2, you can download the pretrained model from here: https://drive.google.com/file/d/1e-b1rYsy97GTWtJ0wbA3uDV8Qqtu339L/view?usp=sharing
1. in fine-tuning.py set the following parameter values:
```
    train_datasets: List[List[str]] = field(default_factory=lambda: [['training_TTT.json', 'output-from-examples-v0']])

    val_dataset: List[str] = field(default_factory=lambda: ['validation_TTT.json', 'output-from-examples-v0'])

    output_dir: str = './output/'

    n_gpus: int = 1    # if using RunPod A40 with 1 GPU like I was
```   
2. python fine-tuning.py to pretrain on the data, let it run until convergence.
3. This will have created folders checkpoint-* under ./output/. Use the last one, rename it to pretrained_model so it doesn't clash with other checkpoint folders from future fine-tuning runs. You can delete the previous checkpoint folders.
4. Now change the following fine-tuning.py parameter values:
```
    adapter_path: Optional[str] = 'output/pretrained_model'
   
    train_datasets: List[List[str]] = field(default_factory=lambda: [['ood_TTT_data1-00000000-train.json', 'output-from-examples-v0']])
   
    val_dataset: List[str] = field(default_factory=lambda: ['ood_TTT_data1-00000009-val.json', 'output-from-examples-v0'])

    eval_steps: int = 10
```   
5. Run this modified fine-tuning.py to produce the task-specific lora adapters. You will have to repeat this for every task instance file! Though you'll probably want to proceed in batches to not run out of disk space from the generated LoRA adapters. To respect the 3-minute time budget, I suggest running this step for no more than 2 minutes 30 seconds each time.
6. python3 merge_lora.py --base_model_path='Qwen/Qwen2-0.5B-Instruct' --lora_path=output/ttft-task1-sample1 --output_path=output/merged_task1_sample1
7. python3 inference.py --model_path output/merged_task1_sample1 --dataset ./ood_TTT_data1-00000000-test.json --output_filepath ./ood_TTT_data1-sample1-solution.json --prompt_version='output-from-examples-v0'
8. open the script grids_equal.py in this repo, and set ground_truth to the test output grid in ood_TTT_data1-00000000-test.json and attempts to the entire content of ood_TTT_data1-sample1-solution.json.
9. Run the grids_equal.py script to find out if this attempt was a success or not.

Note: Chances are you'll want to find a way to automate this entire process, but I didn't.

### LLM-no-TTFT results
Instead of going through steps 1 and 2, you can download the pretrained model from here: https://drive.google.com/file/d/1e-b1rYsy97GTWtJ0wbA3uDV8Qqtu339L/view?usp=sharing
Similar steps to the above, except:
- skip steps 4 and 5, we don't fine-tune on tasks
- step 6 becomes: python3 merge_lora.py --base_model_path='Qwen/Qwen2-0.5B-Instruct' --lora_path=output/pretrained_model --output_path=output/merged_pretrained_model
- step 7 becomes: python3 inference.py --model_path output/merged_pretrained_model --dataset ./ood_TTT_data1-00000000-test.json --output_filepath ./ood_TTT_data1-sample1-solution.json --prompt_version='output-from-examples-v0'

### TTFT+augments results
Instead of going through steps 1 and 2, you can download the pretrained model from here: https://drive.google.com/file/d/1TBGFiuZmZDEbg-9AFuhvYPN1Tizp9Az_/view?usp=sharing
Similar steps to LLM+TTFT, except:

Steps 1 and 2, to pretrain the model on our data from scratch we used a modified fine-tuning.py:
   
```
Instead of (line 384):

    model = AutoModelForCausalLM.from_pretrained(
         model_path,
         quantization_config=bnb_config,
         device_map=get_device_map(n_gpus, model_path, device_map),
         # max_memory={0: '9GB', 1: '8GB'},
         trust_remote_code=True,
         torch_dtype=get_torch_dtype(torch_dtype), #bfloat16 is 4 times slower on Kaggle than float16, on my computer they are the same speed
         attn_implementation=get_flash_attention_implementation(),
         )

we use:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config)

```

and we do not use LoRA for the training:

```
    use_lora: bool = False
    use_rslora = False,
    use_dora = False,

```

At step 4, test-time fine-tuning is done from the fully trained local model, rather than the Qwen online model + the pretraining LoRA adapter, so you can use the normal fine-tuning.py file but set the following parameter values:

```
    model_path: str = 'models/pretrained_from_scratch'
    adapter_path: Optional[str] = None
```

### TTFT-no-augments results (the main TTFT result)
Instead of going through steps 1 and 2, you can download the pretrained model from here: https://drive.google.com/file/d/1Bz9FNc6S6yiVKETMn-1ozRR1I_35d8FM/view?usp=sharing
Starting from the TTFT+augments setup, we must create modified versions of the fine-tuning.py and inference.py files to disable geometric augments. For this you can use the modified scripts found in this repo under the ttft/ folder. Move them to the arc24/scripts folder. Replace data_augmentation.py under arc24/scripts/arc24 with the version in ttft/arc24.

See the full list of steps in LLM+TTFT, but replace with the following:
* for Steps 1 and 2 (pretraining from scratch without augments on our own training data), use: ttft/fine-tuning-pretraining-from-scratch.py
* for the TTFT steps, fine-tune using the script: ttft/fine-tuning-no-augments.py
* Step 6: python3 merge_lora.py --base_model_path='output/pretrained_from_scratch' --lora_path=output/ttft-task1-sample1 --output_path=output/merged_task1_sample1
* for the inference steps, use the script: ttft/inference-no-augments.py

## GridCoder 1 results
Instead of training, you can get the pretrained weights from: https://drive.google.com/file/d/1feeuRxxTwDCfy5hu2Y1-TZGsH4vgpAgu/view?usp=sharing
Name it: model_full.pth
* clone https://github.com/simonouellette35/GridCoder2024
* git checkout experiment/gridcoder2_paper
* in OODGenARC-AGI/generate_toy_data.py, set the VERSION constant to 1
* generate training data using: python generate_toy_data.py
* move these training.json and validation.json files to the GridCoder2024 repo folder.
* inside the GridCoder2024 repo folder: python train_gridcoder2_format.py
* run: python test_gridcoder.py --task alpha_POC --dataset ood_data1.json
* repeat the above line for all tasks from ood_data1 to ood_data7
