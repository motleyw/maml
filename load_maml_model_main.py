import torch
import yaml
import os
from fintune_test import load_maml_model
#from fintune_test import load_data_file_lgi, prepare_dataloader

# Define folder paths
folderpath = ".\maml_lgi_0210"  # Where config.yml and model are stored
test_id = "Sub03"  # ID of the subject testing
device = torch.device("cpu")  # Set device

# Load the MAML model
maml_model = load_maml_model(folderpath, test_id, device)

# Ensure model is explicitly on CPU
maml_model.to(device)

print("MAML successfully loading on CPU")

"""
data_root = "./meta_learn_data_label/Sub03_data"  # Change this to the actual path of your dataset
test_csv_list = [os.path.join(data_root, f"{test_id}.csv")]  # Assuming a CSV file per test ID

# Load data from the CSV file
train_args = yaml.safe_load(open(os.path.join(folderpath, "config.yml"), "r"))["args"]
test_data, test_g_label, test_i_gt, test_loco_label = load_data_file_lgi(
    test_csv_list,
    winlen=train_args["window_length"],
    max_samples=train_args["max_samples"]
)

# Create a DataLoader
test_dataloader = prepare_dataloader(
    test_data, test_g_label, test_i_gt, test_loco_label, device, batch_size=200
)
"""