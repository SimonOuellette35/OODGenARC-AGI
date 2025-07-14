import json
import sys
import re

# Check if a command line argument was provided
if len(sys.argv) != 2:
    print("Usage: python prep_fine-tuning.py <json_file>")
    sys.exit(1)

# Load the JSON file to preprocess
json_file_path = sys.argv[1]
try:
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    print(f"Successfully loaded JSON file: {json_file_path}")
except FileNotFoundError:
    print(f"Error: File '{json_file_path}' not found")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON format in file '{json_file_path}': {e}")
    sys.exit(1)

def format_json_compact_arrays(data):
    """Format JSON with compact arrays but readable overall structure"""
    # First, get the JSON string with proper indentation
    json_str = json.dumps(data, indent=2)
    
    # Use regex to find arrays that span multiple lines and compact them
    # This pattern matches arrays with numbers separated by newlines and whitespace
    def compact_array(match):
        # Extract the array content and remove all whitespace/newlines
        array_content = match.group(0)
        # Find all numbers in the array
        numbers = re.findall(r'\d+', array_content)
        # Return compacted array
        return '[' + ', '.join(numbers) + ']'
    
    # Pattern to match multiline arrays containing only digits
    pattern = r'\[\s*\n\s*(?:\d+(?:\s*,\s*\n\s*\d+)*\s*\n\s*)\]'
    json_str = re.sub(pattern, compact_array, json_str, flags=re.MULTILINE | re.DOTALL)
    
    return json_str

def process_task_sample(task_id, task_sample):
    # Process a single task sample here
    # task_sample will be a dictionary with "train" and "test" keys
    print(f"Processing task with {len(task_sample['train'])} training examples and {len(task_sample['test'])} test examples")
    
    train_file = {}
    val_file = {}
    test_file = {}

    train_file[task_id] = {}
    val_file[task_id] = {}
    test_file[task_id] = {}

    train_file[task_id]['train'] = task_sample['train'][:2]
    val_file[task_id]['train'] = task_sample['train'][2:3]
    test_file[task_id]['train'] = task_sample['train'][:3]

    train_file[task_id]['test'] = task_sample['test'][:1]
    val_file[task_id]['test'] = task_sample['test'][:1]
    test_file[task_id]['test'] = task_sample['test'][1:2]

    # Save train_file to JSON
    base_filename = json_file_path.rsplit('.', 1)[0]  # Remove extension
    train_filename = f"{base_filename}-{task_id}-train.json"
    
    with open(train_filename, 'w') as f:
        formatted_json = format_json_compact_arrays(train_file)
        f.write(formatted_json)
    print(f"Saved training data to: {train_filename}")

    # Save val_file to JSON
    base_filename = json_file_path.rsplit('.', 1)[0]  # Remove extension
    val_filename = f"{base_filename}-{task_id}-val.json"
    
    with open(val_filename, 'w') as f:
        formatted_json = format_json_compact_arrays(val_file)
        f.write(formatted_json)
    print(f"Saved validation data to: {val_filename}")

    # Save test_file to JSON
    base_filename = json_file_path.rsplit('.', 1)[0]  # Remove extension
    test_filename = f"{base_filename}-{task_id}-test.json"
    
    with open(test_filename, 'w') as f:
        formatted_json = format_json_compact_arrays(test_file)
        f.write(formatted_json)
    print(f"Saved test data to: {test_filename}")


# Iterate through each task in the dataset
for task_id, task_sample in data.items():
    print(f"Processing task {task_id}")
    process_task_sample(task_id, task_sample)
