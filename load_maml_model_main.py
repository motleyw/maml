import torch
import yaml
import os
import pandas as pd
import time
from fintune_test import load_maml_model
#from example_read_arduino import *
import queue
import threading
#from fintune_test import load_data_file_lgi, prepare_dataloader
#from main_read_arduino import *
from train_utils import prepare_dataloader
from fintune_test import get_loss
import argparse

latest_window = None
live_spt_batches = []  # Each entry: (data, gait_label, incline_label, loco_label)
incline_label = 0
loco_label = 0


def load_model():
    # Define folder paths
    folderpath = "maml_lgi_0210"  # Where config.yml and model are stored
    test_id = "Sub03"  # ID of the subject testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device

    # Load the MAML model
    maml_model = load_maml_model(folderpath, test_id, device)

    # Ensure model is explicitly on CPU
    maml_model.to(device)

    print("MAML successfully loading on CPU")

    return maml_model, device

def generate_queue(size=100):
    data_queue = queue.Queue(maxsize=size)
    return data_queue

def sensor_reader_csv(csv_path, stop_event, window_size=100):
    """ Reads sensor data from CSV in non-overlapping windows and stores only the latest window. """
    global latest_window  # Declare the global variable
    df = pd.read_csv(csv_path, skiprows=1, header=None, usecols=[1, 2, 4, 5])  # Skip header row
    num_rows = len(df)

    for i in range(window_size, num_rows):  # Read in chunks of 100 rows
        if stop_event.is_set():  # Check stop signal
            print("Stopping sensor thread...")
            break

        if i + window_size > num_rows:  # Stop if there aren't enough rows
            print("Not enough data for a full window, stopping.")
            break

        # Update the latest window with the current 100-row chunk
        latest_window = df.iloc[i-window_size:i].to_numpy(dtype=float)

        time.sleep(0.01)  # Simulate real-time streaming

def start_sensory_reader():
    # Start the sensor reading thread
    csv_path = "./sensor_data.csv"
    stop_event = threading.Event()
    sensor_thread = threading.Thread(target=sensor_reader_csv, args=(csv_path, stop_event, 100), daemon=True)
    sensor_thread.start()
    return stop_event
"""
def start_sensory_reader(data_queue):
    stop_event = threading.Event()
    sensor_thread = threading.Thread(target=read_arduino_data, args=(data_queue, stop_event), daemon=True)
    sensor_thread.start()
"""
def evaluate_model(model, device, stop_event):
    global latest_window
    model.eval()
    with torch.no_grad():
        while not stop_event.is_set():
            if latest_window is not None:
                sensor_data = latest_window

                input_tensor = torch.tensor(sensor_data, dtype=torch.float32).to(device).transpose(0, 1).unsqueeze(0)
                loco_pred, gait_pred, incline_pred = model.net(input_tensor)

                gait_pred_class = gait_pred.argmax(dim=1).cpu().numpy()
                loco_pred_class = loco_pred.argmax(dim=1).cpu().numpy()
                incline_pred_value = incline_pred.squeeze().cpu().numpy()

                print(f"Gait: {gait_pred_class}, Loco: {loco_pred_class}, Incline: {incline_pred_value}")
            else:
                time.sleep(0.5)

def start_model_eval(maml_model, device):
    stop_event = threading.Event()
    eval_thread = threading.Thread(target=evaluate_model, args=(maml_model, device, stop_event), daemon=True)
    eval_thread.start()
    return stop_event

def test_queue(data_queue):
    while True:
        if not data_queue.empty():
            sensor_data = data_queue.get()
            #print(len(sensor_data))
        else:
            time.sleep(0.05)

def add_labeled_window(gait_label, incline_label, loco_label, device):
    global latest_window, live_spt_batches

    if latest_window is not None:
        # Reshape window into correct format
        input_tensor = torch.tensor(latest_window, dtype=torch.float32).to(device).unsqueeze(0).transpose(1, 2)
        gait_tensor = torch.tensor([gait_label], dtype=torch.long).to(device)
        incline_tensor = torch.tensor([incline_label], dtype=torch.float32).to(device)
        loco_tensor = torch.tensor([loco_label], dtype=torch.long).to(device)

        # Save to support batch list
        live_spt_batches.append((input_tensor, gait_tensor, incline_tensor, loco_tensor))

def fine_tune_on_live_data(model, device, optimizer, train_args, updates=1):
    model.train()
    for update in range(updates):
        loss_total = 0
        for (data, gait_gt, incline_gt, loco_gt) in live_spt_batches:
            # Forward
            loco_pred, gait_pred, incline_pred = model.net(data)
            loss = get_loss(gait_pred, incline_pred, loco_pred, gait_gt, incline_gt, loco_gt, train_args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        print(f"[Update {update + 1}] Live fine-tune loss: {loss_total:.4f}")

def get_labels():
    incline_label = input("Input True Incline: ")
    loco_label = input("Input True Locomotion Mode: ")
    return incline_label, loco_label



if __name__ == "__main__":
    maml_model, device = load_model()
    incline_label, loco_label = get_labels()
    stop_event_sensor = start_sensory_reader()
    stop_event_eval = start_model_eval(maml_model, device)
    with open("maml_lgi_0210/config.yml", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    train_args = argparse.Namespace(**config_dict["args"])
    optimizer = optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, maml_model.parameters()),
        lr=train_args.transfer_lr
    )
    try:
        while True:
            if len(live_spt_batches) >= 5:
                fine_tune_on_live_data(maml_model, device, optimizer, train_args, updates=1)
                live_spt_batches.clear()

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
        stop_event_sensor.set()
        stop_event_eval.set()
