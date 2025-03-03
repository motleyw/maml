import torch
import yaml
import os
from fintune_test import load_maml_model
from example_read_arduino import *
import queue
import threading
import torch
#from fintune_test import load_data_file_lgi, prepare_dataloader

def load_model():
    # Define folder paths
    folderpath = ".\maml_lgi_0210"  # Where config.yml and model are stored
    test_id = "Sub03"  # ID of the subject testing
    device = torch.device("cpu")  # Set device

    # Load the MAML model
    maml_model = load_maml_model(folderpath, test_id, device)

    # Ensure model is explicitly on CPU
    maml_model.to(device)

    print("MAML successfully loading on CPU")

    return maml_model

def generate_queue(size=100):
    data_queue = queue.Queue(maxsize=size)
    return data_queue

def start_sensory_reader(data_queue):
    # Start the sensor reading thread
    csv_path = "./meta_learn_data_label/Sub03_data/label_data/0128_Sub03_SA_i33_01.csv"
    stop_event = threading.Event()
    sensor_thread = threading.Thread(target=sensor_reader_csv, args=(csv_path, data_queue, stop_event), daemon=True)
    sensor_thread.start()

def evaluate_model(model, data_queue, device):
    model.eval()
    with torch.no_grad():
        while True:
            if not data_queue.empty():
                sensor_data = data_queue.get()

                # Convert to tensor and reshape as needed
                input_tensor = torch.tensor(sensor_data, dtype=torch.float32).to(device).unsqueeze(0)

                # Forward pass through the model
                loco_pred, gait_pred, incline_pred = model(input_tensor)

                # Extract predictions
                gait_pred_class = gait_pred.argmax(dim=1).cpu().numpy()
                loco_pred_class = loco_pred.argmax(dim=1).cpu().numpy()
                incline_pred_value = incline_pred.squeeze().cpu().numpy()

                print(f"Gait: {gait_pred_class}, Loco: {loco_pred_class}, Incline: {incline_pred_value}")
            else:
                time.sleep(0.05)  # Wait for new data


if __name__ == "__main__":
    maml_model = load_model()
    data_queue = generate_queue()
    start_sensory_reader(data_queue)
    evaluate_model(maml_model, data_queue, torch.device("cpu"))
