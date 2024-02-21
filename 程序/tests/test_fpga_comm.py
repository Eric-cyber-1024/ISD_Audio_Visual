import socket

host = "127.0.0.1"
port = 5004

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setblocking(False)
sock.connect_ex((host,port))

# Create the data to send (32 bytes alternating between 1 and 0)

data = bytes.fromhex('00009001')
print(data)
# Send the data to the server
sock.send(data)





