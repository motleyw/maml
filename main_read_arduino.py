import serial
import csv
import time
from Class_Arduino import Arduino


def read_arduino_data(data_queue, stop_event, window_size=100):
    try:
        arduino = Arduino()  # Instantiate the Arduino class
        time.sleep(2)  # Allow time for Arduino to reset

        print("Reading sensor data. Press Ctrl+C to stop.")
        i = 0
        sliding_window = []
        while not(stop_event.is_set()):
            try:
                arduino.obtain_softsensor_data()
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                data_vector = [arduino.left_sensor_data[1], arduino.left_sensor_data[2], arduino.right_sensor_data[1], arduino.right_sensor_data[2]]
                #print(data_vector)
                if i < window_size:  # Logic to construct 100 len sliding window and add to queue
                    sliding_window.append(data_vector)
                    i += 1
                else:
                    if not data_queue.full():
                        data_queue.put(sliding_window)  # Add data to queue
                        #print("Put data")
                    else:
                        print("Queue full, skipping data.")
                    i = 1
                    sliding_window = [data_vector]
            except KeyboardInterrupt:
                print("Data logging stopped.")
                break
    except serial.SerialException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    read_arduino_data(output_file='sensor_data.csv')