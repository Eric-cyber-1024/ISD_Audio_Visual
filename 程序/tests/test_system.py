# Simple Script to test mic array system
# 
# pop up dialog box to let users to input parameters and then generate a packet to send to FPGA @192.168.1.40 , port = 5004

import os
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
sVersion = '0.6'


class labeledTextbox:
    '''
    a new class for new ttk widget -- label + Entry

    '''
    def __init__(self, master, label_text, enabled=True):
        self.label = ttk.Label(master, text=label_text)
        self.textbox = ttk.Entry(master,state='normal' if enabled else 'disabled')
        self.textbox.insert(tk.END,'0')


class MyDialog:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.title("Pop-up Dialog")

        # Add widgets to the new dialog
        self.label = ttk.Label(self.top, text="This is a pop-up dialog.")
        self.label.pack()

        self.close_button = ttk.Button(self.top, text="Close", command=self.close_popup)
        self.close_button.pack()

    def close_popup(self):
        self.top.destroy()

class manualDelayConfigGUI():
    def __init__(self,parent,micEnableList):
        '''

        micEnableList is a list to ocntrol which mic to be enabled, 0--disable, 1--enable

        '''
        self.top = tk.Toplevel(parent)
        self.micNames=["M{:02d}".format(i) for i in range(1, 33)]
        self.micEnableList = micEnableList

        self.labels_and_textboxes = []
        self.delayValues = []

        # create GUI
        self.create_dialog_box()

    def create_dialog_box(self):
        '''
        generate GUI according to self.micEnableList

        GUI layout will be 4 columns by 8 rows

        1 -- 9 -- 17 -- 25
        .................
        8 -- 16-- 24 -- 32

        **M00 will not be displayed as it's a virtual microphone

        '''
        self.top.title("Manual Delay Configuration")

        
        for i in range(32):
            label_text = self.micNames[i]
            enabled = self.micEnableList[i]  # Replace with your actual list
            pair = labeledTextbox(self.top, label_text, enabled)
            self.labels_and_textboxes.append(pair)

        # Arrange the labeled textboxes in a 4x8 grid
        for i, pair in enumerate(self.labels_and_textboxes):
            row, col = divmod(i, 8)
            pair.label.grid(row=col, column=2*row, sticky='e')
            pair.textbox.grid(row=col, column=2*row+1, sticky='w')

        btnOKPacket = ttk.Button(self.top, text="OK", command=self.fetchParamsFromUI)
        btnOKPacket.grid(row=9)


    def fetchParamsFromUI(self):
        '''
        fetch delays from GUI

        if textboxes are disabled, will return 0

        '''
        
        for pair in self.labels_and_textboxes:
            value = pair.textbox.get()
            try:
                delayVal = float(value)
            except:
                delayVal = 0.0

            self.delayValues.append(delayVal)
        
        
        self.top.destroy()
        print('destroyed',self.delayValues)



class PointSelectionGUI(tk.Frame):
    def __init__(self, master, points, callback):
        super().__init__(master)
        self.master = master
        self.canvas = tk.Canvas(self, width=400, height=400)
        self.canvas.pack()
        self.callback = callback

        # frame_width = 2
        # self.canvas.create_rectangle(
        #     frame_width,
        #     frame_width,
        #     300 - frame_width,
        #     300 - frame_width,
        #     outline='black',
        #     width=frame_width
        # )

        self.points = points
        self.draw_points()
        self.canvas.bind("<Button-1>", self.on_click)

    def draw_points(self):
        for i, (x, y) in enumerate(self.points):
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red")
            if i<10:
                self.canvas.create_text(x, y - 10, text=f"M0{i}", fill="black")
            else:
                self.canvas.create_text(x, y - 10, text=f"M{i}", fill="black")


    def on_click(self, event):
        x, y = event.x, event.y
        for i, (px, py) in enumerate(self.points):
            if abs(px - x) <= 5 and abs(py - y) <= 5:
                #print("Clicked on point index:", i)
                self.callback('%d' %(i))
                break


class paramsDialog:
    def __init__(self):
        self.dialog_box = tk.Tk()
        self.modes = [
            '0: normal', 
            '1: cal',
            '2: cal verify',
            '3: switch mic/output selection',
            '4: turn on BM',
            '5: turn off BM',
            '6: turn on MC',
            '7: turn off MC',
            '8: BLjudge H_CAFFs readback',
            '9: WMcal Wm[] readback',
            
        ]
        
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
        self.hostIP='192.168.1.40'
        self.hostPort=5004
        self.mode = 0
        self.micIndx = 0
        self.micGain = 0
        self.micDisable = 0
        self.setTest = 0
        self.den_out_sel= 0
        self.mc_beta_sel= 0
        self.mc_K_sel   = 0
        self.en_BM_MC_ctrl = 0
        self.offsets= np.array([0,0,0])
        self.srcPos = np.array([0,0,0])
        self.manualDelayConfig = tk.IntVar() # 1-- configure delays manually
        
        self.create_dialog_box()

    # removed,Brian,27 Mar 2024
    # def setUI(self,modeIndx,micIndx,sMicGain,sMicDelay,setTestIndx,sSrcPos):
    #     '''
    #     prepare UI with certain values

    #     mode       -- dropdown list
    #     mic index  -- dropdown list
    #     set test   -- dropdown list
    #     mic gain   -- textbox
    #     mic delay  -- textbox
    #     src pos    -- textbox

    #     '''
    #     self.cbx_micIndx  = 0
    #     self.cbx_testMode = 0
    #     self.tbx_micDelay = sMicDelay
    #     self.tbx_micGain  = sMicGain
    #     self.tbx_srcPos   = sSrcPos
        
    def __str__(self):
        return f"params,{self.hostIP}, {self.hostPort}, {self.mode},{self.micIndx},{self.micGain},{self.setTest},{self.den_out_sel},{self.mc_beta_sel},{self.mc_K_sel},[{self.srcPos[0]},{self.srcPos[1]},{self.srcPos[2]}],[{self.offsets[0]},{self.offsets[1]},{self.offsets[2]}]"


    def printParams(self):
        print(self.hostIP,self.hostPort,self.mode,self.micIndx,self.micGain,self.setTest,self.den_out_sel,self.mc_beta_sel,self.mc_K_sel,self.srcPos,self.offsets)
        # add[save to log as well],Brian,27 Mar 2024
        logger.add_data(self.__str__())

    def fetchParamsFromUI(self):
        
        # Set the values as class properties
        s = self.cbx_mode.get()

        self.hostIP     = self.tbx_hostIP.get()
        self.hostPort   = int(self.tbx_hostPort.get())


        self.mode       = int(s.split(':')[0])
        self.micIndx    = self.micNames.index(self.cbx_micIndx.get())
        self.micGain    = int(self.tbx_micGain.get())
        self.micDisable = int(self.textbox_2.get())

        s = self.cbx_testMode.get()
        self.setTest    = int(s.split(':')[0])
        self.den_out_sel= int(self.tbx_den_out_sel.get())
        self.mc_beta_sel= int(self.tbx_mc_beta_sel.get())
        self.mc_K_sel   = int(self.tbx_mc_K_sel.get())
        self.en_BM_MC_ctrl = int(self.tbx_en_BM_MC_ctrl.get())

        self.srcPos     = np.array(self.tbx_srcPos.get().split(','), dtype=float)
        self.offsets    = np.array(self.tbx_offsets.get().split(','), dtype=float)

        # removed,Brian,27 Mar 2024
        # self.sMode      = self.cbx_mode.get()
        # self.sMicIndx   = self.cbx_micIndx.get()
        # self.sMicGain   = self.tbx_micGain.get()
        # self.sMicDelay  = self.tbx_micDelay.get()
        # self.sSetTest   = self.cbx_testMode.get()
        # self.sSrcPos    = self.tbx_srcPos.get()
        
    
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
        self.den_out_sel= -1
        self.mc_beta_sel= -1
        self.mc_K_sel   = -1
        self.en_BM_MC_ctrl = -1

        self.srcPos     = np.array([-1.0,-1.0,-1.0])
        self.offsets    = np.array([-1.0,-1.0,-1.0])

        self.sMode      = '-1'
        self.sMicIndx   = '-1'
        self.sMicGain   = '-1'
        self.sMicDelay  = '-1'
        self.sSetTest   = '-1'
        self.sSrcPos    = '-1,-1,-1'
        self.manualDelayConfig.set(0)
        
        # Destroy the dialog box
        #self.dialog_box.destroy()

    def showInfo(self,sMsg):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lbl_info.config(text='%s, %s' %(timestamp,sMsg))

    def send_message(self, message):
        # Process the message in the main application
        # change cbx_micIndx according to message
        if message!='0':
            self.cbx_micIndx.current(int(message)-1)
        else:
            messagebox.showinfo(title="Alert",message='M00 does not exist actually!')

    def sendPacket(self):
        global logger
        # clear lbl_info first
        self.showInfo('')
        sendBuf=b'SET0'
        
        self.fetchParamsFromUI()
        self.printParams()
    
        #Z=distance between camera and object, x is left+/right-, y is down+/up-
        this_location=[6, 0.2, 0.3]

        # revised[add offsets],Brian,18 Mar 2024
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
        message10 = int(self.den_out_sel) # den_out_sel, previously micDelay

        # revise[added message11, message12],Brian, 27 Mar 2024
        message11 = int(self.mc_beta_sel) # mc_beta_sel
        message12 = int(self.mc_K_sel)    # mc_K_sel

        # revise[added message13],Brian, 28 Mar 2024
        message13 = int(self.en_BM_MC_ctrl) # en_BM_MC_ctrl
 
        # check to configure delays manually or not
        if self.manualDelayConfig.get()==1:
            # pop up UI to fill in delay values manually
            micEnableList = [1 if i in (0, 8, 16, 24) else 0 for i in range(32)]
            delayConfigGUI = manualDelayConfigGUI(self.dialog_box,micEnableList)
            # for pop up GUI to end
            self.dialog_box.wait_window(delayConfigGUI.top)
            print(delayConfigGUI.delayValues)
        else:
            # get delays based on formula and send packet to FPGA
            _,refDelay,_ = delay_calculation(self.srcPos,self.offsets[0],self.offsets[1],self.offsets[2])   
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
                
            sendBuf=bytes([message1,message2,message3,message4,message5,message6,message7,message8,message9,message10,message11,message12,message13])
                
            # append packet to sendBuf
            sendBuf += packet
            
            logger.add_data('data,%s,%s,%s' %(bytes(sendBuf),np.array2string(refDelay),np.array2string(self.srcPos)))


            if send_and_receive_packet(self.hostIP,self.hostPort,sendBuf,timeout=3):
                print('data transmission ok')
                self.showInfo('tx ok')
                logger.add_data('tx ok')
            else:
                print('data transmission failed')
                logger.add_data('tx failed')

    
    
    
    def create_dialog_box(self):

        # Set the title of the dialog box
        self.dialog_box.title("Set Parameters--v"+sVersion)


        lbl_hostIP = ttk.Label(self.dialog_box, text="Host IP")
        lbl_hostIP.pack()
        self.tbx_hostIP = ttk.Entry(self.dialog_box)
        self.tbx_hostIP.insert(0,'192.168.1.40')
        self.tbx_hostIP.pack()

        lbl_hostPort = ttk.Label(self.dialog_box, text="Host Port")
        lbl_hostPort.pack()
        self.tbx_hostPort = ttk.Entry(self.dialog_box)
        self.tbx_hostPort.insert(0,'5004')
        self.tbx_hostPort.pack()


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

        # revised micDelay to den_out_sel
        # added widgets for mc_beta_sel, mc_K_sel

        lbl_den_out_sel = ttk.Label(self.dialog_box, text="den_out_sel")
        lbl_den_out_sel.pack()
        self.tbx_den_out_sel = ttk.Entry(self.dialog_box)
        self.tbx_den_out_sel.insert(0,'8')
        self.tbx_den_out_sel.pack()

        lbl_mc_beta_sel = ttk.Label(self.dialog_box, text="mc_beta_sel")
        lbl_mc_beta_sel.pack()
        self.tbx_mc_beta_sel = ttk.Entry(self.dialog_box)
        self.tbx_mc_beta_sel.insert(0,'4')
        self.tbx_mc_beta_sel.pack()

        lbl_mc_K_sel = ttk.Label(self.dialog_box, text="mc_K_sel")
        lbl_mc_K_sel.pack()
        self.tbx_mc_K_sel = ttk.Entry(self.dialog_box)
        self.tbx_mc_K_sel.insert(0,'0')
        self.tbx_mc_K_sel.pack()


        lbl_en_BM_MC_ctrl = ttk.Label(self.dialog_box, text = 'en_BM_MC_ctrl')
        lbl_en_BM_MC_ctrl.pack()
        self.tbx_en_BM_MC_ctrl = ttk.Entry(self.dialog_box)
        self.tbx_en_BM_MC_ctrl.insert(0,'0')
        self.tbx_en_BM_MC_ctrl.pack()

        lbl_srcPos = ttk.Label(self.dialog_box, text="source pos")
        lbl_srcPos.pack()
        self.tbx_srcPos = ttk.Entry(self.dialog_box)
        self.tbx_srcPos.insert(0,'0,0,0')
        self.tbx_srcPos.pack()

        lbl_offsets = ttk.Label(self.dialog_box, text="x,y,z Offsets")
        lbl_offsets.pack()
        self.tbx_offsets = ttk.Entry(self.dialog_box)
        self.tbx_offsets.insert(0,'0,0,0')
        self.tbx_offsets.pack()
        


        self.lbl_info   = ttk.Label(self.dialog_box,text='')
        self.lbl_info.pack()

        # revise[removed ok, cancel buttons],Brian,15 Mar 2024
        # # Create the buttons
        # ok_button = ttk.Button(self.dialog_box, text="OK", command=self.get_user_inputs)
        # ok_button.pack(side=tk.LEFT)

        # cancel_button = ttk.Button(self.dialog_box, text="Cancel", command=self.cancel)
        # cancel_button.pack(side=tk.LEFT)

        


        # Create a list of points
        pts = getMicPositions(0,0,0)
        points = [(pt[0]*500+150,150-pt[1]*500) for pt in pts]

        # # Create the PointSelectionGUI and embed it in the main window
        self.point_selection = PointSelectionGUI(self.dialog_box, points,self.send_message)
        self.point_selection.pack(side=tk.LEFT, padx=10, pady=10)

        # Add checkbox to have manual delay configuration or not
        ckbx_ManualDelayConfig = ttk.Checkbutton(self.dialog_box,text='Manual Delay Config',
                                                 onvalue=1,offvalue=0,
                                                 variable=self.manualDelayConfig)
        
        ckbx_ManualDelayConfig.pack()
        self.manualDelayConfig.set(0) # not checked by default
        
        btnSendPacket = ttk.Button(self.dialog_box, text="Send Packet", command=self.sendPacket)
        btnSendPacket.pack(side=tk.LEFT)

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

    # Check if the folder exists
    if not os.path.exists('log'):
        # Create the folder
        os.makedirs('log')
        print("log folder created successfully.")
    else:
        print("log folder already exists.")

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