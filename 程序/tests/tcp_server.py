import socket
from struct import *
import ctypes


class struct_packet(ctypes.BigEndianStructure):
    '''

    The Generic 32=bit Packet Structure

    '''
    _fields_ = [
        ("R/W",       ctypes.c_uint32, 2),  # 2 bit wide
        ("Reserved1", ctypes.c_uint32, 2),  # 2 bit wide
        ("Types",     ctypes.c_uint32, 2),  # 2 bit wide
        ("Mode",      ctypes.c_uint32, 1),  # 1 bit wide
        ("Reserved2", ctypes.c_uint32, 5),  # 5 bit wide
        ("Command",   ctypes.c_uint32, 16), # 16 bits wide
        ("Reserved3", ctypes.c_uint32, 4)   # 4 bits wide
    ]


class struct_cmd_adc(ctypes.BigEndianStructure):
    '''

    Structure for 'ADC Related' Commands

    '''
    _fields_ = [
        ("type",       ctypes.c_uint16, 3),  # 3 bit wide, 4--channel gain, 6--digital volume, 7--lpf selection
        ("mic#",       ctypes.c_uint16, 5),  # 5 bit wide, mic index 0-31
        ("value",      ctypes.c_uint16, 8),  # 8 bit wide, value
    ]

tPacket = struct_packet()
tPacket2= struct_packet()
tCmd    = struct_cmd_adc()

def parseDataBytes(data):
    unpacked_value, = unpack_from('!i',data)
    pack_into('!i',tPacket,0,unpacked_value)
    pack_into('!h',tCmd,0,tPacket.Command)
    
    for field_name, field_type, _ in tPacket._fields_:
        field_value = getattr(tPacket, field_name)
        print(field_name + ":", hex(field_value))

    for field_name, field_type, _ in tCmd._fields_:
        field_value = getattr(tCmd,field_name)
        print(field_name + ':', hex(field_value))


# Define the server's IP address and port
SERVER_IP  = "127.0.0.1"
SERVER_PORT= 5004

# Create a TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the IP address and port
server_socket.bind((SERVER_IP,SERVER_PORT))

# Listen for incoming connections
server_socket.listen()

print(f"Server is listening on {SERVER_IP}:{SERVER_PORT}")

while True:
    # Accept a client connection
    client_socket, client_address = server_socket.accept()
    
    print(f"Client connected: {client_address}")
    
    # Receive data from the client
    data = client_socket.recv(1024)
    
    
    # parse data bytes
    parseDataBytes(data)

    
    # Send a response back to the client
    #response = "Hello from the server!"
    #client_socket.send(response.encode())
    
    # Close the client socket
    client_socket.close()