# Simple Script to test mic array system
# 
# pop up dialog box to let users to input parameters and then generate a packet to send to FPGA @192.168.1.40 , port = 5004

import sys
import socket
import selectors
import types
import time
# wx add
import math
import numpy as np ###### require install and adjust to certain edition 1.13.3
from test_delay_cal import *
import tkinter as tk
from tkinter import ttk, messagebox

sel = selectors.DefaultSelector()
sendBuf=b'SET0'
RecvBuf=[]
sendFlag=False
waitFlag=True
MIC_NUMBER=32
INDEX =[x for x in range (MIC_NUMBER )]


class paramsDialog:
    def __init__(self):
        self.dialog_box = tk.Tk()
        
        self.dropdown_1 = None
        self.dropdown_2 = None
        self.textbox_1 = None
        self.textbox_2 = None
        self.textbox_3 = None
        self.textbox_4 = None

        self.modes = ['0: normal', 
                      '1: cal',
                      '2: cal verify',
                      '3: switch mic/output selection',
                      '4: turn on BM',
                      '5: turn off BM',
                      '6: turn on MC',
                      '7: turn off MC']
        
        self.testModes=['0: PS_enMC-0,PS_enBM-0,FFTgain: 2, MIC8-data_bm_n, MIC9-ifftout',
                        '4: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_n, MIC9-ifftout',
                        '8: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_n, MIC9-ifftout, delta_t=0',
                        '9: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_ym_n, MIC9-data_fbf_d_MC',
                        '10: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_ym_n, MIC9-data_fbf_d_MC, delta_t=0',
                        '11: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_n_dly, MIC9-data_bm_0',
                        '12: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_n_dly, MIC9-data_bm_0, delta_t=0',
                        '13: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_0, MIC9-data_ym_0']
        

        # set default values of the properties
        self.mode = 0
        self.micIndx = 0
        self.micGain = 0
        self.micDisable = 0
        self.setTest = 0
        self.micDelay = 0
        self.srcPos = np.array([0,0,0])
        
        self.create_dialog_box()

    
    def printParams(self):
        print(self.mode,self.micIndx,self.micGain,self.setTest,self.micDelay,self.srcPos)

    def fetchParamsFromUI(self):
        s = self.cbx_mode.get()
        
        
        
        # Set the values as class properties
        self.mode       = int(s.split(':')[0])
        self.micIndx    = int(self.cbx_micIndx.get())
        self.micGain    = int(self.textbox_1.get())
        self.micDisable = int(self.textbox_2.get())

        s = self.cbx_testMode.get()
        self.setTest    = int(s.split(':')[0])
        self.micDelay   = int(self.tbx_micDelay.get())

        self.srcPos     = np.array(self.tbx_srcPos.get().split(','), dtype=float)
    
    def get_user_inputs(self):
        # Retrieve the selected values from the dropdown lists and textboxes

        self.fetchParamsFromUI()

        # Destroy the dialog box
        self.dialog_box.destroy()
    
    def cancel(self):
        # Set all values to -1
        self.dropdown_value_1 = self.dropdown_value_2 = self.textbox_value_1 = self.textbox_value_2 = self.textbox_value_3 = self.textbox_value_4 = -1
        
        # Destroy the dialog box
        self.dialog_box.destroy()


    def sendPacket(self):
        
        self.fetchParamsFromUI()
        self.printParams()
        #messagebox.showinfo("Infor", "blah blah blah...")
    
    def create_dialog_box(self):

        # Set the title of the dialog box
        self.dialog_box.title("Set Parameters")

        # Create labels for the dropdown lists
        lbl_mode = ttk.Label(self.dialog_box, text="Mode:")
        lbl_mode.pack()

        # Create the first dropdown list
        self.cbx_mode = ttk.Combobox(self.dialog_box, values=self.modes)
        self.cbx_mode.pack()

        lbl_micIndx = ttk.Label(self.dialog_box, text="Mic#:")
        lbl_micIndx.pack()

        # Create the second dropdown list
        self.cbx_micIndx = ttk.Combobox(self.dialog_box, values=[i for i in range(32)])
        self.cbx_micIndx.pack()

        # Create the textboxes with labels
        label_3 = ttk.Label(self.dialog_box, text="mic gain")
        label_3.pack()
        self.textbox_1 = ttk.Entry(self.dialog_box)
        self.textbox_1.pack()

        label_4 = ttk.Label(self.dialog_box, text="mic disable")
        label_4.pack()
        self.textbox_2 = ttk.Entry(self.dialog_box)
        self.textbox_2.pack()

        label_5 = ttk.Label(self.dialog_box, text="set test")
        label_5.pack()
        
        longest_text_2 = max([str(value) for value in self.testModes], key=len)
        self.cbx_testMode = ttk.Combobox(self.dialog_box, values=self.testModes, width=len(longest_text_2))
        self.cbx_testMode.pack()

        lbl_micDelay = ttk.Label(self.dialog_box, text="mic delay")
        lbl_micDelay.pack()
        self.tbx_micDelay = ttk.Entry(self.dialog_box)
        self.tbx_micDelay.pack()

        lbl_srcPos = ttk.Label(self.dialog_box, text="source pos")
        lbl_srcPos.pack()
        self.tbx_srcPos = ttk.Entry(self.dialog_box)
        self.tbx_srcPos.pack()

        # Create the buttons
        ok_button = ttk.Button(self.dialog_box, text="OK", command=self.get_user_inputs)
        ok_button.pack(side=tk.LEFT)

        cancel_button = ttk.Button(self.dialog_box, text="Cancel", command=self.cancel)
        cancel_button.pack(side=tk.LEFT)

        btnSendPacket = ttk.Button(self.dialog_box, text="Send Packet", command=self.sendPacket)
        btnSendPacket.pack(side=tk.LEFT)

        # Start the main event loop
        self.dialog_box.mainloop()




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

def delay_calculation_v1(thisposition):
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


if __name__ == '__main__':

    params = paramsDialog()
    params.printParams()
    
    #Z=distance between camera and object, x is left+/right-, y is down+/up-
    this_location=[6, 0.2, 0.3]
    delay=delay_calculation_v1(this_location)
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
    print(message)
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


    


    # if len(sys.argv) != 9:
    #     print("usage:", sys.argv[0], "<host> <port> <mode> <mic> <mic_gain> <mic_disable> <set_test> <mic_delay>")
    #     sys.exit(1)
    # host, port, mode, mic, mic_vol, mic_disable, set_test,mic_delay = sys.argv[1:9]

    host = '192.168.1.40'
    port = 5004

    

    waitFlag = True
    while True:
        try:
            while True:
                # wait for signal to leave this wait loop
                inStr = input("Please input 'start' to send:")
                if inStr=='start':
                    break
            
            # get parameters from User Interface
            mode        = params.mode
            mic         = params.micIndx
            mic_gain    = params.micGain
            mic_disable = params.micDisable
            set_test    = params.setTest
            mic_delay   = params.micDelay

            message5  = int(mode)        # mode
            message6  = int(mic)         # mic
            message7  = int(mic_gain)    # mic_gain
            message8  = int(mic_disable) # mic_disable
            message9  = int(set_test)    # set_test
            message10 = int(mic_delay)   # mic_delay

            sel = selectors.DefaultSelector()     #wx add can work looply
            start_connections(host, int(port))
    #     global sendFlag,sendBuf
            #org sendBuf=bytes([message1,message2,message3,message4])
            #sendBuf=bytes([message1,message2,message3,message4,message5,message6])
            
            

            # test,Brian,05 Mar 2024
            # payload = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
            #    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,0x1f,0x20]

            
            _,refDelay,_ = delay_calculation(params.srcPos)   
            refDelay = refDelay*48e3
            refDelay = np.max(refDelay)-refDelay
            refDelay = np.round(refDelay)

            #convert refDelay to byte
            #but make sure that they are within 0 to 255 first!!
            assert (refDelay>=0).all() and (refDelay<=255).all()

            # payload = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
            #         0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,0x1f,0x20]

            refDelay = refDelay.astype(np.uint8)
            payload = refDelay.tobytes()
            print('refDelay',refDelay)
            print('payload',payload)
            print('sendBuf',sendBuf)

            packet = prepareMicDelaysPacket(payload)
            if validateMicDelaysPacket(packet):
                print('packet ok')
            else:
                print('packet not ok')
            
            sendBuf=bytes([message1,message2,message3,message4,message5,message6,message7,message8,message9,message10])
            
            # append packet to sendBuf
            sendBuf += packet
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


