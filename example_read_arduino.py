import time
import pandas as pd
import numpy as np

def sensor_reader_csv(csv_path, data_queue, stop_event, window_size=100):
    """ Reads sensor data from a CSV file line by line and adds it to the queue. """
    df = pd.read_csv(csv_path, skiprows=1, header=None)  # Load CSV file into a DataFrame
    num_rows = len(df)

    for i in range(0, num_rows, window_size):  # Iterate through each row
        print(i)
        if stop_event.is_set():  # Check if stop signal is given
            print("Stopping sensor thread...")
            break

        if i + window_size > num_rows:
            print("Out of data")
            break

        window_data = df.iloc[i:i+window_size].to_numpy(dtype=float)

        if not data_queue.full():
            data_queue.put(window_data)  # Add full window to queue
            print(f"Window {i//window_size + 1} added to queue")
        else:
            print("Queue full, skipping data.")

        time.sleep(1)  # Simulate real-time streaming