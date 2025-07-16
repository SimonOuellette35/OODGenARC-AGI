import json
from datasets.toy_data_generator import generate_data
from model.transformer_model import StandardTransformerModel


trainN = 200000
valN = 1000
model = StandardTransformerModel.instantiate_from_config_file('gridcoder_cfg.json')
VERSION = 2


# Generate training data
print(f"Generating {trainN} training samples...")
input_sequences, programs = generate_data(model, num_samples=trainN, version=VERSION)

# Convert to list of dictionaries for JSON serialization
training_data = []
for i in range(trainN):
    sample = {
        "input_sequence": input_sequences[i].tolist(),
        "prog": programs[i]
    }
    training_data.append(sample)

# Save training data to JSON file
with open("training.json", "w") as f:
    json.dump(training_data, f)

print("Training data saved to training.json")

# Generate validation data
print(f"Generating {valN} validation samples...")
input_sequences, programs = generate_data(model, num_samples=valN)

# Convert to list of dictionaries for JSON serialization
validation_data = []
for i in range(valN):
    sample = {
        "input_sequence": input_sequences[i].tolist(),
        "prog": programs[i]
    }
    validation_data.append(sample)

# Save validation data to JSON file
with open("validation.json", "w") as f:
    json.dump(validation_data, f)

print("Validation data saved to validation.json")
