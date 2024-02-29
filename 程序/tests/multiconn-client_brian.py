#!/usr/bin/env python3

import sys
import socket
import selectors
import types
# wx add
import math
import numpy as np ###### require install and adjust to certain edition 1.13.3
import serial
import threading
import datetime
from data_logger import DataLogger

sel = selectors.DefaultSelector()
sendBuf=b'SET0'
RecvBuf=[]
sendFlag=False
MIC_NUMBER=32
INDEX =[x for x in range (MIC_NUMBER )]


# Function to read data from the serial port
def read_serial_data(ser):
    while True:
        line = ser.readline().decode('ISO-8859-1').strip()
        #print(line)
        
        if line.find('Read after trigger')>=0:
            print(line)
            logger.add_data(line)
        #    break

def start_connections(host, port):
    server_addr = (host, port)
    print("starting connection to ",  server_addr)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(server_addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    data = types.SimpleNamespace(
            outb=b"",
    )
    sel.register(sock, events,data=data)

def service_connection(key, mask):
    sock = key.fileobj
    data = key.data
    if mask & selectors.EVENT_READ:
        recv_data = sock.recv(1024)  # Should be ready to read
        if recv_data:
            print("received", repr(recv_data) )
            print("closing connection")
            sel.unregister(sock)
            sock.close()
        elif not recv_data:
            print("closing connection")
            sel.unregister(sock)
            sock.close()
    if mask & selectors.EVENT_WRITE:
        global sendFlag
        if sendFlag==True:      
            data.outb = sendBuf
            print("sending", repr(data.outb), "to connection", host, port)
            sent = sock.send(data.outb)  # Should be ready to write
            sendFlag=False

def struct_packet(RW_field,mode,mic_gain,mic_num,en_bm,en_bc,delay_binary_output_x,mic_en,type,reserved):
     #convert into binary form 
#     MIC_change = np.asarray(list(map(int,bin (int(MIC_change))[2:].zfill(4))))
#     MIC_change = [0,1,1,1]
     this_mic_no  = decToBin(mic_num,5)
     this_type=decToBin(type,2)
     this_reserved=decToBin(reserved,4)
#     MC_BM_packet = BM_MC_status(BM_MC_bit_received)
#     delay_time   = delay_binary_output
#     mic_ONOFF = mic_onoff_ID
 #    Channel = np.asarray(list(map(int,bin (int(choice))[2:].zfill(2)))) # register change ch num and turn into binary
     #putting them into array order 
     return np.hstack((RW_field,mode,mic_gain,this_mic_no,en_bm,en_bc,delay_binary_output_x,mic_en,this_type,this_reserved))


# create mic ID in Binary form 
def decToBin(this_value,bin_num):
    num = bin(this_value)[2:].zfill(bin_num)
    num = list(map(int,num)) # convert list str to int 
    return num 

def BintoINT(Binary):
    integer = 0 
    result_hex = 0 
    for x in range(len(Binary)):
        integer = integer + Binary[x]*2**(len(Binary)-x-1)
    result_hex = hex(integer)
    return result_hex

if __name__ == '__main__':

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = DataLogger(log_interval=1, file_path="log/%s_fpga.log" %(timestamp))  # Specify the data file path
    # Start the logging process
    logger.start_logging()
    print('data logger started')
    logger.add_data('ISD Microphone Array System (FPGA Communication) Started')



    # Configure the serial port
    ser = serial.Serial('COM15', 115200)

    # Start a separate thread to read data from the serial port
    serial_thread = threading.Thread(target=read_serial_data, args=(ser,), daemon=True)
    serial_thread.start()



    RW_field=[1,1]
    mode=0
    mic_gain=[1,0]
    mic_num=0
    en_bm=1
    en_bc=1
    mic_en=1
    type=0
    reserved=0
    dummy = [0,0,0,0,1,0,0,1,1,0,1,0,0]
    message=struct_packet(RW_field,mode,mic_gain,mic_num,en_bm,en_bc,dummy,mic_en,type,reserved)
    #print(message)
    messagehex = BintoINT(message)
    print(messagehex)
    message1 = int(messagehex[2:4],16) # hex at  1 and 2  
    message2 = int(messagehex[4:6],16) # hex at  3 and 4 
    message3 = int(messagehex[6:8],16)  # hex at  5 and 6 
    message4 = int(messagehex[8:],16)
    print("m1:{},m2:{},m3:{},m4:{}\n".format(message1,message2,message3,message4))
    #org if len(sys.argv) != 3:
    #org    print("usage:", sys.argv[0], "<host> <port> ")
    #org   sys.exit(1)
    #org host, port = sys.argv[1:3]

    #if len(sys.argv) != 5:
    #    print("usage:", sys.argv[0], "<host> <port> <mode> <mic>")
    #    sys.exit(1)
    #host, port, mode, mic = sys.argv[1:5]

    if len(sys.argv) != 7:
        print("usage:", sys.argv[0], "<host> <port> <mode> <mic> <mic_vol> <mic_disable>")
        sys.exit(1)
    host, port, mode, mic, mic_vol, mic_disable = sys.argv[1:7]

    message5 = int(sys.argv[3])
    message6 = int(sys.argv[4])
    message7 = int(sys.argv[5])
    message8 = int(sys.argv[6])

    while True:
        try:
            while True:
                inStr = input("Please input 'start' to send:")
                if inStr=='start':
                    break
                elif inStr=='getparams':
                    print([message1,message2,message3,message4,message5,message6,message7,message8]) 
                    pass
                elif inStr.find('n')==0:
                    sMsg = '****** Selected mic index=%s' %(inStr[1:])
                    print(sMsg)
                    logger.add_data(sMsg)
                    message6=int(inStr[1:])
                    break
            
            sel = selectors.DefaultSelector()     #wx add can work looply
            start_connections(host, int(port))

            print([message1,message2,message3,message4,message5,message6,message7,message8])
    #     global sendFlag,sendBuf
            #org sendBuf=bytes([message1,message2,message3,message4])
            #sendBuf=bytes([message1,message2,message3,message4,message5,message6])
            sendBuf=bytes([message1,message2,message3,message4,message5,message6,message7,message8])
            sendFlag=True
            while True:
                events = sel.select(timeout=None)
                if events:
                    for key, mask in events:
                        service_connection(key, mask)
                # Check for a socket being monitored to continue.
                if not sel.get_map():
                    print("exit 2")
                    break
        except KeyboardInterrupt:
            print("caught keyboard interrupt, exiting")
            # stop data logger
            logger.stop_logging()
            sys.exit(1)
        finally:
            print("exit 3")
            sel.close()
            



