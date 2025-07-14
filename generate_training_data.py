import json
import numpy as np
from datasets.data_generator import generate
from model.transformer_model import StandardTransformerModel


def numpy_to_python(obj):
    """Convert numpy types to native Python types recursively."""
    if isinstance(obj, np.ndarray):
        return numpy_to_python(obj.tolist())
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                          np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return complex(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    else:
        return obj


trainN = 400000
valN = 1000
model = StandardTransformerModel.instantiate_from_config_file('gridcoder_cfg.json')


# Generate training data
print(f"Generating {trainN} training samples...")
input_sequences, programs = generate(model, num_samples=trainN)

# Convert to list of dictionaries for JSON serialization
training_data = []
for i in range(trainN):
    sample = {
        "input_sequence": numpy_to_python(input_sequences[i]),
        "prog": numpy_to_python(programs[i])
    }
    training_data.append(sample)

# Save training data to JSON file
with open("training.json", "w") as f:
    json.dump(training_data, f)

print("Training data saved to training.json")

# Generate validation data
print(f"Generating {valN} validation samples...")
input_sequences, programs = generate(model, num_samples=valN)

# Convert to list of dictionaries for JSON serialization
validation_data = []
for i in range(valN):
    sample = {
        "input_sequence": numpy_to_python(input_sequences[i]),
        "prog": numpy_to_python(programs[i])
    }
    validation_data.append(sample)

# Save validation data to JSON file
with open("validation.json", "w") as f:
    json.dump(validation_data, f)

print("Validation data saved to validation.json")