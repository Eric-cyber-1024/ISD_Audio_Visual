#!/usr/bin/env python3

import sys
import socket
import selectors
import types
# wx add
import math
import numpy as np ###### require install and adjust to certain edition 1.13.3

sel = selectors.DefaultSelector()
sendBuf=b'SET0'
RecvBuf=[]
sendFlag=False
MIC_NUMBER=32
INDEX =[x for x in range (MIC_NUMBER )]

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

def get_position():
 mic_position = (                        #  mic_array_position_21Jul2023.csv
                [ 0, 0.05,  0],      #0 
                [ 0, 0.0353553390593274,  0.0353553390593274],  #1 
                [ 0, 0,  0.05], #2
                [ 0, -0.0353553390593274,  0.0353553390593274], #3
                [ 0, -0.05,  0], #4
                [ 0, -0.0353553390593274, -0.0353553390593274],  #5
                [ 0, 0,  -0.05],  #6
                [ 0,  0.0353553390593274, -0.0353553390593274], #7     
                [ 0,  0.0923879532511287, 0.038268343236509], #8  
                [ 0, 0.038268343236509,  0.0923879532511287],#9
                [ 0, -0.038268343236509,  0.0923879532511287],#10
                [ 0, -0.0923879532511287,  0.038268343236509],#11
                [ 0, -0.0923879532511287,  -0.038268343236509], #12
                [ 0, -0.038268343236509,  -0.0923879532511287],#13
                [ 0, 0.038268343236509,  -0.0923879532511287],#14
                [ 0, 0.0923879532511287,  -0.038268343236509],#15
                [ 0, 0.106066017177982,  0.106066017177982],#16
                [ 0, 0.0574025148547635,  0.138581929876693], #17
                [ 0, 0,  0.15],#18
                [ 0, -0.0574025148547635,  0.138581929876693],#19
                [ 0, -0.106066017177982,  0.106066017177982],#20
                [ 0, -0.138581929876693,  0.0574025148547635],#21
                [ 0, -0.15,  0],#22
                [ 0, -0.138581929876693,  -0.0574025148547635],#23
                [ 0, -0.106066017177982,  -0.106066017177982],#24
                [ 0, -0.0574025148547635,  -0.138581929876693], #25
                [ 0, 0,  -0.15],#26
                [ 0, 0.0574025148547635,  -0.138581929876693],#27
                [ 0, 0.106066017177982,  -0.106066017177982],#28
                [ 0, 0.138581929876693,  -0.0574025148547635],#29
                [ 0, 0.15,  0], #30
                [ 0, 0.138581929876693,  0.0574025148547635]#31
                )

 mic_position=np.array(mic_position)
 return mic_position

def delay_calculation(thisposition):
    c_detail = thisposition     #this_location=[6, 0.2, 0.3]
    c_detail = np.array(c_detail)
    Targetposition=c_detail

    SPEED_OF_SOUND   =340.29 
    mic_position = get_position() # getting mic position 
    mic_ref_ori = mic_position[2] # center mic  the top one from inner circle
    soure_ref = Targetposition - mic_ref_ori
    magnitude_s2p = [0] * MIC_NUMBER 
      # np.linalg.norm(求范数)
  #linalg=linear（线性）+algebra（代数），norm则表示范数
  #  https://blog.csdn.net/hqh131360239/article/details/79061535
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

def BintoINT(Binary):
  integer = 0 
  result_hex = 0 
  for x in range(len(Binary)):
    integer = integer + Binary[x]*2**(len(Binary)-x-1)
  result_hex = hex(integer)
  return result_hex

#Z=distance between camera and object, x is left+/right-, y is down+/up-
this_location=[6, 0.2, 0.3]
delay=delay_calculation(this_location)
print(delay)
#converting the delay into binary format 
delay_binary_output = delay_to_binary(delay)
#print(delay_binary_output)
#need to do later
RW_field=[1,1]
mode=0
mic_gain=[1,0]
mic_num=0
en_bm=1
en_bc=1
mic_en=1
type=0
reserved=0
message=struct_packet(RW_field,mode,mic_gain,mic_num,en_bm,en_bc,delay_binary_output[0],mic_en,type,reserved)
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

if len(sys.argv) != 9:
    print("usage:", sys.argv[0], "<host> <port> <mode> <mic> <mic_gain> <mic_disable> <set_test> <mic_delay")
    sys.exit(1)
host, port, mode, mic, mic_vol, mic_disable, set_test,mic_delay = sys.argv[1:9]

message5 = int(sys.argv[3])
message6 = int(sys.argv[4])
message7 = int(sys.argv[5])
message8 = int(sys.argv[6])
message9 = int(sys.argv[7])
message10 = int(sys.argv[8])

while True:
    try:
#org        while(input("Please input 'start' to send:")!='start' ):
#org            pass
        while True:
            inStr = input("Please input 'start' to send:")
            if inStr=='start':
                break
            elif inStr.find('n')==0:
                message6=int(inStr[1:])
                break
        
        sel = selectors.DefaultSelector()     #wx add can work looply
        start_connections(host, int(port))
   #     global sendFlag,sendBuf
        #org sendBuf=bytes([message1,message2,message3,message4])
        #sendBuf=bytes([message1,message2,message3,message4,message5,message6])
        sendBuf=bytes([message1,message2,message3,message4,message5,message6,message7,message8,message9,message10])
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
        sys.exit(1)
    finally:
        print("exit 3")
        sel.close()

