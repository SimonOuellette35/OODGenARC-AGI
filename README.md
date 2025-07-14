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
