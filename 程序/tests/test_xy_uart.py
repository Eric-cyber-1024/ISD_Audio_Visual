import csv
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
if len(sys.argv) == 3:
    index = int(sys.argv[2])  

    # Read the table from the CSV file
    table = []
    try:
        with open('locs.csv', 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                x, y = map(int, row)
                table.append((x, y))
    except FileNotFoundError:
        print("CSV file not found.")
        sys.exit(1)
    except csv.Error:
        print("Error reading CSV file.")
        sys.exit(1)

    # Validate the index
    if not (0 <= index < len(table)):
        print("Invalid index provided.")
        sys.exit(1)

    # Extract x and y values from the table based on the index
    x, y = table[index]

    print('index=%d' %(index))
    print('x,y=%d,%d' %(x,y))


elif len(sys.argv) == 4:
    x = int(sys.argv[2])
    y = int(sys.argv[3])

else:
    print("Usage: python test_xy_uart.py <com_port> <index> OR python test_xy_uart.py <com_port> <x> <y>")
    sys.exit(1)

com_port = sys.argv[1]

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

# Wait for the serial thread to finish
serial_thread.join()

# Close the serial port
ser.close()