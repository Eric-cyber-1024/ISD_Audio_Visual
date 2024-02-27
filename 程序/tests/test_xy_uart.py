import serial
import threading
import sys

# Function to read data from the serial port
def read_serial_data(ser):
    while True:
        line = ser.readline().decode().strip()
        print(line)
        if line == "Enter position x and y":
            break

# Get command-line arguments
if len(sys.argv) != 4:
    print("Usage: python serial_client.py <com_port> <x> <y>")
    sys.exit(1)

com_port = sys.argv[1]
x = int(sys.argv[2])
y = int(sys.argv[3])

# Configure the serial port
ser = serial.Serial(com_port, 9600)

# Construct the string
message = f"{x} {y}"

# Send the string to the Arduino
ser.write(message.encode())

# Start a separate thread to read data from the serial port
serial_thread = threading.Thread(target=read_serial_data, args=(ser,), daemon=True)
serial_thread.start()

# Main thread continues execution
# You can perform other tasks here if needed

# Wait for the serial thread to finish (optional)
serial_thread.join()

# Close the serial port
ser.close()