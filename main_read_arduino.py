import serial
import csv
import time
from Class_Arduino import Arduino


def read_arduino_data(output_file='sensor_data.csv'):
    try:
        arduino = Arduino()  # Instantiate the Arduino class
        time.sleep(2)  # Allow time for Arduino to reset
        
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Left Sensor Time", "Left Hip", "Left Knee", "Right Sensor Time", "Right Hip", "Right Knee"])  # Header row
            
            print("Reading sensor data. Press Ctrl+C to stop.")
            while True:
                try:
                    arduino.obtain_softsensor_data()
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"{arduino.left_sensor_data[0]}: {arduino.left_sensor_data}, {arduino.right_sensor_data}")
                    writer.writerow(arduino.left_sensor_data + arduino.right_sensor_data)
                except KeyboardInterrupt:
                    print("Data logging stopped.")
                    break
    except serial.SerialException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    read_arduino_data(output_file='sensor_data.csv')