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
2. Move the OOD_TTT_data*.json files to the arc24/scripts folder
3. From OODGenARC-AGI repo, run: python generate_TTT_data.py (to generate fine-tuning data for the TTFT model)
4. Move the generated training_TTT.json and validation_TTT.json files over to arc24/scripts
5. Under arc24/ run pip install -r requirements.txt
6. Some issues I faced:
   a) had to install: pip install torch==2.5.0 transformers==4.52.1
   b) had to install: pip install vllm
   c) when running inference.py I had a weird exception that I resolved by doing: pip install --upgrade vllm
7. Copy the modified scripts from this repo's ttft/ folder to the arc24/scripts folder above.

### LLM+TTFT results
1. python fine-tuning.py to pretrain on the data
2. Use fine-tuning-ttft.py (with adapter_path=output of previous operation) to produce the task-specific lora adapter. (set the config in the file)
3. python3 merge_lora.py --base_model_path='Qwen/Qwen2-0.5B-Instruct' --lora_path=models/ttft-task6-sample1 --output_path=output/merged_task1_sample1
4. python3 inference.py --model_path output/merged_task1_sample1 --dataset ./ood_TTT_data1-sample1-test.json --output_filepath ./ood_TTT_data1-sample1-solution.json --prompt_version='output-from-examples-v0'

### LLM-no-TTFT results
1. python fine-tuning.py to pretrain on the data
2. python3 merge_lora.py --base_model_path='Qwen/Qwen2-0.5B-Instruct' --lora_path=models/ttft-task6-sample1 --output_path=output/merged_task1_sample1
3. python3 inference.py --model_path output/merged_task1_sample1 --dataset ./ood_TTT_data1-sample1-test.json --output_filepath ./ood_TTT_data1-sample1-solution.json --prompt_version='output-from-examples-v0'

### TTFT+augments results
1. python fine-tuning-no-weights.py to pretrain on the data
2. Use fine-tuning-ttft.py (with adapter_path=output of previous operation) to produce the task-specific lora adapter. (set the config in the file)
3. python3 merge_lora.py --base_model_path='Qwen/Qwen2-0.5B-Instruct' --lora_path=models/ttft-task6-sample1 --output_path=output/merged_task1_sample1
4. python3 inference.py --model_path output/merged_task1_sample1 --dataset ./ood_TTT_data1-sample1-test.json --output_filepath ./ood_TTT_data1-sample1-solution.json --prompt_version='output-from-examples-v0'

### TTFT-no-augments results (the main TTFT result)
1. python fine-tuning-ood.py to pretrain on the data
2. Use fine-tuning-ttft-no-augments.py (with adapter_path=output of previous operation) to produce the task-specific lora adapter. (set the config in the file)
3. python3 merge_lora.py --base_model_path='Qwen/Qwen2-0.5B-Instruct' --lora_path=models/ttft-task6-sample1 --output_path=output/merged_task1_sample1
4. python3 inference-no-augments.py --model_path output/merged_task1_sample1 --dataset ./ood_TTT_data1-sample1-test.json --output_filepath ./ood_TTT_data1-sample1-solution.json --prompt_version='output-from-examples-v0'



