import socket
import selectors
import types
from Mic_function import *

HOST = "192.168.1.40"
PORT = 5004


def delay_calculation_v1(thisposition):
    c_detail = thisposition     #this_location=[6, 0.2, 0.3]
    c_detail = np.array(c_detail)
    Targetposition=c_detail

    SPEED_OF_SOUND   =340.29 
    mic_position = get_position() # getting mic position 
    mic_ref_ori = mic_position[2] # center mic  the top one from inner circle
    soure_ref = Targetposition - mic_ref_ori
    magnitude_s2p = [0] * MIC_NUMBER 
    magnitude_s2r = np.linalg.norm(soure_ref, axis=0, keepdims=True)
    for x in INDEX:
        magnitude_s2p[x] = Targetposition - mic_position[x]
        magnitude_s2p[x] = np.linalg.norm(magnitude_s2p[x], axis=0, keepdims=True)
    delay = [0]*MIC_NUMBER 
    for x in INDEX:
        delay[x] = - (magnitude_s2r-magnitude_s2p[x])/SPEED_OF_SOUND
    delay = abs(min(delay))+delay 
    delay = -1*( delay - max(delay))    #big to small reverse
    delay_phase = delay * 48000   ##   48KHz
    delay =(np.round(delay / 1e-6, 6)) 
   # mic_delay_extra = find_calibration_value(Vision.distance,Vision.x,Vision.y)
    for x in INDEX:
        delay[x] =(delay[x] ) * 1e-6
        delay_phase[x] = delay[x]*48000
        delay_phase[x] = int(delay_phase[x]*360/512*256)   # 512 FFT window
    minimum= abs(min(delay_phase))
    delay_phase = delay_phase + minimum
    delay_phase    = np.reshape(delay_phase,MIC_NUMBER ) 
    return delay_phase

#return delay integer to binary 
def delay_to_binary(delay):
    delay_binary_output = [0] * len(delay) 
    for x in INDEX:
        delay_binary_output[x] = np.asarray( list(map(int, bin (int(delay[x]))[2:].zfill(13))) )
    return delay_binary_output  

# create mic ID in Binary form 
def decToBin(this_value,bin_num):
    num = bin(this_value)[2:].zfill(bin_num)
    num = list(map(int,num)) # convert list str to int 
    return num 

def struct_packet(RW_field,mode,mic_gain,mic_num,en_bm,en_bc,delay_binary_output_x,mic_en,type,reserved):
     #convert into binary form 
     this_mic_no  = decToBin(mic_num,5)
     this_type=decToBin(type,2)
     this_reserved=decToBin(reserved,4)
     #putting them into array order 
     return np.hstack((RW_field,mode,mic_gain,this_mic_no,en_bm,en_bc,delay_binary_output_x,mic_en,this_type,this_reserved))

def BintoINT(Binary):
    integer = 0 
    result_hex = 0 
    for x in range(len(Binary)):
        integer = integer + Binary[x]*2**(len(Binary)-x-1)
    result_hex = hex(integer)
    return result_hex



def send_and_receive_packet(host, port, packet, timeout=1):
    # Create a socket object
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Set the timeout
    sock.settimeout(timeout)

    try:
        sock.connect((host, port))  # connect to the server
        # Send the packet
        sock.send(packet)

        # Wait for the return packet
        data = sock.recv(1024)

        # Check if the received data matches the expected return packet
        if data == b'copy':
            return True
        else:
            return False

    except socket.timeout:
        print('socket timeout!!')
        return False

    finally:
        # Close the socket
        sock.close()


# add,Brian,05 Mar 2024
def prepareMicDelaysPacket(payload):
    '''
    prepare packet for mic delays

    payload -- list of bytes to be sent 

    return packet = payload + checksum (xor all bytes in payload)

    '''
    checksum = 0
    for byte in payload:
        checksum ^= byte
    packet = bytes(payload) + bytes([checksum])

    return packet

# add,Brian,05 Mar 2024
def validateMicDelaysPacket(packet):
    payload = packet[:-1]
    checksum = packet[-1]

    calculated_checksum = 0
    for byte in payload:
        calculated_checksum ^= byte

    if checksum == calculated_checksum:
        return True
    else:
        return False



def start_connections(host, port):
    server_addr = (host, port)
    print("starting connection to", server_addr)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(server_addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    data = types.SimpleNamespace(outb=b"")
    sel = selectors.DefaultSelector()  # Vincent added from main
    sel.register(sock, events, data=data)
    return sel


def service_connection(key, mask, sel, host, port):
    sock = key.fileobj
    data = key.data

    if mask & selectors.EVENT_READ:
        recv_data = sock.recv(1024)  # shd be ready to read
        if recv_data:
            print("received", repr(recv_data))
            print("closing connection")
            sel.unregister(sock)
            sock.close()
        elif not recv_data:
            print("closing connection")
            sel.unregister(sock)
            sock.close()

    if mask & selectors.EVENT_WRITE:
        global sendFlag
        # TODO change to #if sendFlag == True?
        if sendFlag:
            data.outb = sendBuf
            print("sending", repr(data.outb), "to connection", host, port)
            sent = sock.send(data.outb)  # shd be ready to write
            sendFlag = False


def create_and_send_packet(host, port, message):
    while True:
        try:
            while (input("Please input 'start' to send:") != 'start'):
                pass
            sel = start_connections(host, port)
            global sendBuf, sendFlag
            sendBuf = message
            sendFlag = True
            while True:
                events = sel.select(timeout=None)
                if events:
                    for key, mask in events:
                        service_connection(key, mask, sel, host, port)
        # WX: Check for a socket being monitored to continue
                if not sel.get_map():
                    print("exit 2")
                break
        except KeyboardInterrupt:
            print("caught keyboard interrupt, exiting")
            sys.exit(1)
        finally:
            print("exit 3")
            sel.close()






