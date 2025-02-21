import os
import sys
import serial
import serial.tools.list_ports

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)

class Arduino:

    def __init__(self):
        self.left_sensor_data = [0, 0, 0]
        self.right_sensor_data = [0, 0, 0]
        print('connecting to arduino')
        self.arduino = self._connectToArduino()  # connect to the ardunio

    def _connectToArduino(self):
        # Directly specify the port
        arduino_port = "/dev/ttyACM0"  # Change the port to match your Arduino
        try:
            print(f"Attempting to connect to {arduino_port}")
            arduino = serial.Serial(arduino_port, baudrate=250000, timeout=1)
            arduino.flush()
            arduino.reset_input_buffer()
            print(f"Connected successfully to {arduino_port}")
            return arduino
        except serial.SerialException as e:
            raise IOError(f"Unable to connect to {arduino_port}: {e}")
            
    def obtain_softsensor_data(self):
        soft_sensor = self.arduino
        while soft_sensor.inWaiting() == 0:
            pass
        try:
            data = soft_sensor.readline()
            dataarray = data.decode().rstrip().split("\t")
            # soft_sensor.reset_input_buffer()
            # print('dataarray'+str(dataarray))
            self.left_sensor_data[0] = float(dataarray[0])  # sensor time
            self.left_sensor_data[1] = float(dataarray[1])  # raw value of the textile sensor 1, usually left hip
            self.left_sensor_data[2] = float(dataarray[2])  # raw value of the textile sensor 2, usually left knee

            self.right_sensor_data[0] = float(dataarray[0])  # sensor time
            self.right_sensor_data[1] = float(dataarray[3])  # raw value of the textile sensor 3, usually right hip
            self.right_sensor_data[2] = float(dataarray[4])  # raw value of the textile sensor 4, usually right knee
            # return rawdata1, rawdata2, rawdata3, rawdata4
        except (KeyboardInterrupt, SystemExit, IndexError, ValueError):
            pass