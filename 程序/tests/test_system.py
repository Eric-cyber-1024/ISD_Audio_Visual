# Simple Script to test mic array system
# 
# pop up dialog box to let users to input parameters and then generate a packet to send to FPGA @192.168.1.40 , port = 5004

import sys
import socket
import selectors
import types
import time, datetime
# wx add
import math
import numpy as np ###### require install and adjust to certain edition 1.13.3
from test_delay_cal import *
import tkinter as tk
from tkinter import ttk, messagebox
from data_logger import DataLogger

sel = selectors.DefaultSelector()
sendBuf=b'SET0'
RecvBuf=[]
sendFlag=False
waitFlag=True
MIC_NUMBER=32
HOST_NAME ='192.168.1.40'
PORT      =5004
INDEX =[x for x in range (MIC_NUMBER )]



class PointSelectionGUI(tk.Frame):
    def __init__(self, master, points, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.master = master
        self.canvas = tk.Canvas(self, width=400, height=400)
        self.canvas.pack()
        self.points = points
        self.draw_points()
        self.canvas.bind("<Button-1>", self.on_click)

    def draw_points(self):
        for i, (x, y) in enumerate(self.points):
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red")
            self.canvas.create_text(x, y - 10, text=f"{i}", fill="black")

    def on_click(self, event):
        x, y = event.x, event.y
        for i, (px, py) in enumerate(self.points):
            if abs(px - x) <= 5 and abs(py - y) <= 5:
                print("Clicked on point index:", i)
                break


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
        
        self.micNames=["M{:02d}".format(i) for i in range(1, 33)]

        self.testModes=[
            '0: PS_enMC-0,PS_enBM-0,FFTgain: 2, MIC8-data_bm_n, MIC9-ifftout',
            '4: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_n, MIC9-ifftout',
            '8: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_n, MIC9-ifftout, delta_t=0',
            '9: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_n, MIC9-data_fbf_d_MC',
            '10: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_ym_n, MIC9-data_fbf_d_MC, delta_t=0',
            '11: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_n_dly, MIC9-data_bm_0',
            '12: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_n_dly, MIC9-data_bm_0, delta_t=0',
            '13: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_bm_0, MIC9-data_ym_0',
            '14: PS_enMC-0,PS_enBM-0,FFTgain: 5, MIC8-data_ym_judge, MIC9-data_bm_0'
        ]
        

        # set default values of the properties
        self.mode = 0
        self.micIndx = 0
        self.micGain = 0
        self.micDisable = 0
        self.setTest = 0
        self.micDelay = 0
        self.srcPos = np.array([0,0,0])
        
        self.create_dialog_box()

    def setUI(self,modeIndx,micIndx,sMicGain,sMicDelay,setTestIndx,sSrcPos):
        '''
        prepare UI with certain values

        mode       -- dropdown list
        mic index  -- dropdown list
        set test   -- dropdown list
        mic gain   -- textbox
        mic delay  -- textbox
        src pos    -- textbox

        '''
        self.cbx_micIndx  = 0
        self.cbx_testMode = 0
        self.tbx_micDelay = sMicDelay
        self.tbx_micGain  = sMicGain
        self.tbx_srcPos   = sSrcPos

    def printParams(self):
        print(self.mode,self.micIndx,self.micGain,self.setTest,self.micDelay,self.srcPos)

    def fetchParamsFromUI(self):
        
        # Set the values as class properties
        s = self.cbx_mode.get()
        self.mode       = int(s.split(':')[0])
        self.micIndx    = self.micNames.index(self.cbx_micIndx.get())
        self.micGain    = int(self.tbx_micGain.get())
        self.micDisable = int(self.textbox_2.get())

        s = self.cbx_testMode.get()
        self.setTest    = int(s.split(':')[0])
        self.micDelay   = int(self.tbx_micDelay.get())

        self.srcPos     = np.array(self.tbx_srcPos.get().split(','), dtype=float)

        self.sMode      = self.cbx_mode.get()
        self.sMicIndx   = self.cbx_micIndx.get()
        self.sMicGain   = self.tbx_micGain.get()
        self.sMicDelay  = self.tbx_micDelay.get()
        self.sSetTest   = self.cbx_testMode.get()
        self.sSrcPos    = self.tbx_srcPos.get()
    
    def get_user_inputs(self):
        # Retrieve the selected values from the dropdown lists and textboxes

        self.fetchParamsFromUI()

        # Destroy the dialog box
        #self.dialog_box.destroy()
    
    def cancel(self):
        # Set all values to -1
        self.mode       = -1
        self.micIndx    = -1
        self.micGain    = -1
        self.micDisable = -1

        self.setTest    = -1
        self.micDelay   = -1

        self.srcPos     = np.array([-1.0,-1.0,-1.0])

        self.sMode      = '-1'
        self.sMicIndx   = '-1'
        self.sMicGain   = '-1'
        self.sMicDelay  = '-1'
        self.sSetTest   = '-1'
        self.sSrcPos    = '-1,-1,-1'
        # Destroy the dialog box
        #self.dialog_box.destroy()

    def showInfo(self,sMsg):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lbl_info.config(text='%s, %s' %(timestamp,sMsg))

    def sendPacket(self):
        global logger
        # clear lbl_info first
        self.showInfo('')
        sendBuf=b'SET0'
        
        self.fetchParamsFromUI()
        self.printParams()
    
        #Z=distance between camera and object, x is left+/right-, y is down+/up-
        this_location=[6, 0.2, 0.3]
        delay=delay_calculation_v1(this_location)
        print(delay)
        logger.add_data('%s,%s' %('delay',np.array2string(delay)))
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
    

        message5  = int(self.mode)        # mode
        message6  = int(self.micIndx)     # mic
        message7  = int(self.micGain)     # mic_gain
        message8  = int(self.micDisable)  # mic_disable
        message9  = int(self.setTest)     # set_test
        message10 = int(self.micDelay)    # mic_delay
 
        
        _,refDelay,_ = delay_calculation(self.srcPos)   
        refDelay = refDelay*48e3
        refDelay = np.max(refDelay)-refDelay
        refDelay = np.round(refDelay)

        #convert refDelay to byte
        #but make sure that they are within 0 to 255 first!!
        assert (refDelay>=0).all() and (refDelay<=255).all()

            
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
        
        logger.add_data('data,%s,%s,%s' %(bytes(sendBuf),np.array2string(refDelay),np.array2string(self.srcPos)))


        if send_and_receive_packet(HOST_NAME,PORT,sendBuf,timeout=3):
            print('data transmission ok')
            self.showInfo('tx ok')
            logger.add_data('tx ok')
        else:
            print('data transmission failed')
            logger.add_data('tx failed')
    
    def create_dialog_box(self):

        # Set the title of the dialog box
        self.dialog_box.title("Set Parameters")

        # Create labels for the dropdown lists
        lbl_mode = ttk.Label(self.dialog_box, text="Mode:")
        lbl_mode.pack()

        # Create the first dropdown list
        self.cbx_mode = ttk.Combobox(self.dialog_box, values=self.modes)
        self.cbx_mode.current(0)
        self.cbx_mode.pack()

        lbl_micIndx = ttk.Label(self.dialog_box, text="Mic#:")
        lbl_micIndx.pack()

        # Create the second dropdown list
        self.cbx_micIndx = ttk.Combobox(self.dialog_box, values=self.micNames)
        self.cbx_micIndx.current(0)
        self.cbx_micIndx.pack()

        # Create the textboxes with labels
        label_3 = ttk.Label(self.dialog_box, text="mic gain")
        label_3.pack()
        self.tbx_micGain = ttk.Entry(self.dialog_box)
        self.tbx_micGain.insert(0,'10')
        self.tbx_micGain.pack()

        label_4 = ttk.Label(self.dialog_box, text="mic disable")
        label_4.pack()
        self.textbox_2 = ttk.Entry(self.dialog_box)
        self.textbox_2.insert(0,'30')
        self.textbox_2.pack()

        label_5 = ttk.Label(self.dialog_box, text="set test")
        label_5.pack()
        
        longest_text_2 = max([str(value) for value in self.testModes], key=len)
        self.cbx_testMode = ttk.Combobox(self.dialog_box, values=self.testModes, width=len(longest_text_2))
        self.cbx_testMode.current(3)
        self.cbx_testMode.pack()

        lbl_micDelay = ttk.Label(self.dialog_box, text="mic delay")
        lbl_micDelay.pack()
        self.tbx_micDelay = ttk.Entry(self.dialog_box)
        self.tbx_micDelay.insert(0,'0')
        self.tbx_micDelay.pack()

        lbl_srcPos = ttk.Label(self.dialog_box, text="source pos")
        lbl_srcPos.pack()
        self.tbx_srcPos = ttk.Entry(self.dialog_box)
        self.tbx_srcPos.insert(0,'0,0,0')
        self.tbx_srcPos.pack()

        self.lbl_info   = ttk.Label(self.dialog_box,text='')
        self.lbl_info.pack()

        # revise[removed ok, cancel buttons],Brian,15 Mar 2024
        # # Create the buttons
        # ok_button = ttk.Button(self.dialog_box, text="OK", command=self.get_user_inputs)
        # ok_button.pack(side=tk.LEFT)

        # cancel_button = ttk.Button(self.dialog_box, text="Cancel", command=self.cancel)
        # cancel_button.pack(side=tk.LEFT)

        btnSendPacket = ttk.Button(self.dialog_box, text="Send Packet", command=self.sendPacket)
        btnSendPacket.pack(side=tk.LEFT)


        # Create a list of points
        # pts = getMicPositions(0,0.5,0)
        # points = [(pt[0]*500,pt[1]*500) for pt in pts]

        # # Create the PointSelectionGUI and embed it in the main window
        # point_selection = PointSelectionGUI(self.dialog_box, points)
        # point_selection.pack(side=tk.LEFT, padx=10, pady=10)

        # Start the main event loop
        self.dialog_box.mainloop()


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

    # initialize logger
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = DataLogger(log_interval=1, file_path="log/%s_sys.log" %(timestamp))  # Specify the data file path
    logger.start_logging()
    
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
    
    params = paramsDialog()
    params.printParams()

    logger.stop_logging()
    exit()