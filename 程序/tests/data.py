import pandas as pd
import datetime
import re
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pylab as plt
from test_delay_cal import *
import yaml
import math
import os




def selectLogFile(initialPath,logType):
    # Define the default file extensions
    fpgaLogType = [
        ("FPGA Log Files", "*_fpga.log"),
    ]

    cameraLogType = [
        ("Camera Log Files", "*_data.log"),
    ]

    # Open the file selection dialog
    if logType=='fpga':
        file_path = askopenfilename(initialdir=initialPath,filetypes=fpgaLogType)
    elif logType=='camera':
        file_path = askopenfilename(initialdir=initialPath,filetypes=cameraLogType)

    # Show the selected file path
    # print("Selected File:", file_path)

    return file_path


def fpgaLogToDf(filePath):
    data = []
    timestamp_pattern = re.compile(r'^(\d{8}_\d{6})')
    index_pattern = re.compile(r'Selected mic index=(\d+)')
    data_pattern = re.compile(r'RxBuf=([A-F0-9]{2,4})')

    with open(filePath, 'r') as file:
        lines = file.readlines()
        line_count = len(lines)

        i = 0
        packetIndx=0
        while i < line_count:
            line = lines[i].strip()

            timestamp_match = timestamp_pattern.search(line)
            if timestamp_match:
                

                index_match = index_pattern.search(line)
                if index_match:
                    mic_indx = int(index_match.group(1))
                    for j in range(i + 1, min(i + 9, line_count)):
                        timestamp_match = timestamp_pattern.search(lines[j])
                        timestamp = timestamp_match.group(1)
                        data_line = lines[j].strip()
                        data_match = data_pattern.search(data_line)
                        if data_match:
                            hex_string = data_match.group(1)
                            if len(hex_string) == 2:
                                decimal_value = 0
                            elif len(hex_string) == 3:
                                # since it's a single digit, it should be positive
                                decimal_value = int(hex_string[0], 16)
                            else:
                                # need to check if decimal_value >127, if so, convert it to negative
                                decimal_value = int(hex_string[:2], 16)
                                if decimal_value>127:
                                    decimal_value-=256

                            data.append([timestamp,packetIndx,mic_indx, decimal_value])
                        elif data_line.find('RxBuf=0')>0:
                            data.append([timestamp,packetIndx,mic_indx, -21])

                    packetIndx+=1

                

            i += 1

    df = pd.DataFrame(data, columns=['Timestamp','packetIndex', 'index', 'data'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'],format="%Y%m%d_%H%M%S")
    return df
    

def logToDf(filepath):

    cam_dict = {}
    dcam_dict = {}
    index = -1
    list_idx = 0

    try: 
        with open(filepath) as log:
            for line in log:
                if "index#" in line:
                    index = line.split("#")[1].strip()
                
                if index != -1:
                    if (" cam" in line):
                        line_splitted = re.sub("[() ]", "", line)
                        line_splitted = line_splitted.strip().split(",")
                        
                        timestamp = line_splitted[0].split(":")[0]
                        data1 = [timestamp, index] + line_splitted[1:]
                        cam_dict[list_idx] = data1

                    elif (" dcam" in line):
                        line_splitted = re.sub("[() ]", "", line)
                        line_splitted = line_splitted.strip().split(",")

                        timestamp = line_splitted[0].split(":")[0]
                        data2 = [timestamp, index] + line_splitted[1:]
                        dcam_dict[list_idx] = data2
                        list_idx += 1

        df_cam = pd.DataFrame.from_dict(cam_dict, orient="index", 
                                        columns=["Timestamp", "index", "xv", "yv", "zv", "xi", "yi", "roll", "pitch", "yaw"])

        df_dcam = pd.DataFrame.from_dict(dcam_dict, orient="index",
                                        columns=["Timestamp", "index", "yd", "xd", "zd"])
    
        # Add[convert some columns to numeric and datetime],Brian,02 Mar 2024
        columns_to_convert          = ["index", "xv", "yv", "zv", "xi", "yi", "roll", "pitch", "yaw"]
        df_cam[columns_to_convert]  = df_cam[columns_to_convert].apply(pd.to_numeric)
        df_cam['Timestamp']         = pd.to_datetime(df_cam['Timestamp'], format="%Y%m%d_%H%M%S")
        columns_to_convert          = ['index',"yd", "xd", "zd"]
        df_dcam[columns_to_convert] = df_dcam[columns_to_convert].apply(pd.to_numeric)
        df_dcam['Timestamp']        = pd.to_datetime(df_dcam['Timestamp'], format="%Y%m%d_%H%M%S")
    except:
        print("File Not Found")
        return pd.DataFrame(), pd.DataFrame()
    
    return df_cam, df_dcam


def loadCalibrationData(yamlFileName):
    with open(yamlFileName, 'r') as file:
        data = yaml.safe_load(file)
    # Extract the camera matrix and distortion coefficients
    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['dist_coeff'])
    #print(camera_matrix,dist_coeffs)
    return camera_matrix,dist_coeffs


def plotDelay(talkbox_locs, df_cam_summary, df_fpga_summary, ):

   
    for i in range(len(talkbox_locs)):

        # get talkbox location name
        talkbox_loc_name = talkbox_locs[i]

        # step 1. from df_cam_summary['end_timestamp'] and look for corresponding data from df_fpga_summary
        cameraStartTime = df_cam_summary['start_timestamp'][i]
        cameraEndTime   = df_cam_summary['end_timestamp'][i]
        startIndx = np.argmax(df_fpga_summary['start_timestamp'] > cameraStartTime)
        endIndx = np.where(df_fpga_summary['end_timestamp']   < cameraEndTime)[0][-1]
        

        micIndices = df_fpga_summary['mic index'][startIndx:endIndx+1]
        fpgaDelays = df_fpga_summary['average_delay'][startIndx:endIndx+1]

        
        # Filter the indices and data arrays between 0 and 29
        filtered_indices = micIndices[(micIndices >= 0) & (micIndices <= 29)]
        filtered_data    = fpgaDelays[(micIndices >= 0) & (micIndices <= 29)]

        # Get the sorted indices
        sorted_indices = np.argsort(filtered_indices)

        # Rearrange the data based on the sorted indices
        fpgaDelays = filtered_data[sorted_indices]

        vec = df_cam_summary['average_loc'][i][:3]
        vec[:2]=vec[:2]*-1
        #vec[1]*=-1.
        
        print("vec: ", vec,talkbox_loc_name)
        #vec[0]-=0.1
        #vec[1]-=0.09
        _,refDelay,mic_names = delay_calculation(vec,0,0.5,0)
        
        # we consider only 2:2+29 only
        mic_names= mic_names[2:2+29]
        refDelay = refDelay[2:2+29]*48e3

        plt.figure(figsize=(10,6))

        # subplot #1
        plt.subplot(211)
        plt.plot(mic_names,np.round(refDelay),'+-')
        plt.plot(mic_names,np.round(fpgaDelays),'*-')
        plt.title('talkbox pos # %d' %(talkbox_loc_name))
        
        #plt.xlabel('mic name')
        combined = np.concatenate((refDelay, fpgaDelays))
        min_value = np.min(combined)
        max_value = np.max(combined)
        plt.yticks(range(int(min_value)-1,int(max_value)+1))
        plt.xticks(range(len(mic_names)), mic_names, fontsize=10)
        plt.grid(True)
        plt.ylabel('delay/samples')
        plt.legend(['ref','actual'])
        
        # subplot #2, showing differences
        plt.subplot(212)

        diff = np.round(refDelay)-np.round(fpgaDelays)
        plt.plot(mic_names,diff,'+-')
        plt.title('talkbox pos # %d' %(talkbox_loc_name))
        
        plt.yticks(range(int(np.min(diff)),int(np.max(diff))))
        plt.xticks(range(len(mic_names)), mic_names, fontsize=10)
        plt.grid(True)
        plt.ylabel('delay/samples')
        plt.legend(['ref-actual'])

        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()

        isOutputDirExist = os.path.exists("output")
        if not isOutputDirExist:
            os.makedirs("output")

        plt.savefig('output/%s.png' %(talkbox_loc_name))


def plotDelayFar(talkbox_loc_name, df_cam_summary, df_fpga_summary):
    vectors_uv = [row[3:5] for row in df_cam_summary['average_loc']]

    for i in range(6):

        # get talkbox location name
        talkbox_loc_name = talkbox_locs[i]

        # step 1. from df_cam_summary['end_timestamp'] and look for corresponding data from df_fpga_summary
        cameraStartTime = df_cam_summary['start_timestamp'][i]
        cameraEndTime   = df_cam_summary['end_timestamp'][i]
        startIndx = np.argmax(df_fpga_summary['start_timestamp'] > cameraStartTime)
        endIndx = np.where(df_fpga_summary['end_timestamp']   < cameraEndTime)[0][-1]
        

        micIndices = df_fpga_summary['mic index'][startIndx:endIndx+1]
        fpgaDelays = df_fpga_summary['average_delay'][startIndx:endIndx+1]

        
        # Filter the indices and data arrays between 0 and 29
        filtered_indices = micIndices[(micIndices >= 0) & (micIndices <= 29)]
        filtered_data    = fpgaDelays[(micIndices >= 0) & (micIndices <= 29)]
        print("fpgaDelays: ", fpgaDelays)
        # Get the sorted indices
        sorted_indices = np.argsort(filtered_indices)

        # Rearrange the data based on the sorted indices
        fpgaDelays = filtered_data[sorted_indices]
        
        vec_uv_c_normalized, refDelayFar, mic_names = delay_calculation_far(vectors_uv[i])

        mic_names = mic_names[2:2+29]
        refDelayFar = refDelayFar[2:2+29]*48e3

        f1 = plt.figure(figsize=(10,6))
        plt.subplot(211)
        plt.plot(mic_names, refDelayFar, "+-")
        plt.plot(mic_names, fpgaDelays, "*-")
        plt.title("talkbox pos # %d" %(talkbox_loc_name))
        plt.xlabel('mic name')

        plt.xticks(range(len(mic_names)), mic_names, fontsize=10)
        plt.grid(True)
        plt.ylabel('delay/samples')
        plt.legend(['refFar','actual'])

        # subplot #2, showing differences
        plt.subplot(212)
        diff = np.round(refDelayFar)-np.round(fpgaDelays)
        plt.plot(mic_names,diff,'+-')
        plt.title('talkbox pos # %d' %(talkbox_loc_name))
        
        plt.yticks(range(int(np.min(diff)),int(np.max(diff))))
        plt.xticks(range(len(mic_names)), mic_names, fontsize=10)
        plt.grid(True)
        plt.ylabel('delay/samples')
        plt.legend(['refFar-actual'])

        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()

        isOutputDirExist = os.path.exists("output")
        if not isOutputDirExist:
            os.makedirs("output")

        plt.savefig('output/far_%s.png' %(talkbox_loc_name))


def plotTimeDelayError(talkbox_locs, df_cam_summary, df_fpga_summary):

    srcCoordinatesWList = np.zeros((6, 3))
    srcCoordinatesWList[0] = np.array([38.00, -33.00, 367.00])
    srcCoordinatesWList[1] = np.array([24.00, -23.00, 367.00])
    srcCoordinatesWList[2] = np.array([-45.00, -23.00, 364.00])
    srcCoordinatesWList[3] = np.array([24.00, 22.00, 371.00])
    srcCoordinatesWList[4] = np.array([-46.00, 23.00, 371.00])
    srcCoordinatesWList[5] = np.array([-10.00, 0.00, 370.00])

    for i in range(len(talkbox_locs)):

        # get talkbox location name
        talkbox_loc_name = talkbox_locs[i]

        # step 1. from df_cam_summary['end_timestamp'] and look for corresponding data from df_fpga_summary
        cameraStartTime = df_cam_summary['start_timestamp'][i]
        cameraEndTime   = df_cam_summary['end_timestamp'][i]
        startIndx = np.argmax(df_fpga_summary['start_timestamp'] > cameraStartTime)
        endIndx = np.where(df_fpga_summary['end_timestamp']   < cameraEndTime)[0][-1]
        

        micIndices = df_fpga_summary['mic index'][startIndx:endIndx+1]
        fpgaDelays = df_fpga_summary['average_delay'][startIndx:endIndx+1]

        
        # Filter the indices and data arrays between 0 and 29
        filtered_indices = micIndices[(micIndices >= 0) & (micIndices <= 29)]
        filtered_data    = fpgaDelays[(micIndices >= 0) & (micIndices <= 29)]

        # Get the sorted indices
        sorted_indices = np.argsort(filtered_indices)

        # Rearrange the data based on the sorted indices
        fpgaDelays = filtered_data[sorted_indices]

        vec = df_cam_summary['average_loc'][i][:3]
        vec[:2]=vec[:2]*-1
        #vec[0]-=0.1
        #vec[1]-=0.09

        vec[2]*=100.

        _, refDelay1, mic_names = delay_calculation(vec)
        refDelay2, _ = delay_calculation_eq2(vec)

        # we consider only 2:2+29 only
        mic_names= mic_names[2:2+29]
        refDelay1 = refDelay1[2:2+29]*48e3
        refDelay2 = refDelay2[2:2+29]*48e3


        # # subplot #1
        f1 = plt.figure(figsize=(10,6))
        plt.subplot(211)
        plt.plot(mic_names,refDelay1,'+-')
        plt.plot(mic_names,refDelay2,'*-')
        plt.title('talkbox pos # %d' %(talkbox_loc_name))
        
        #plt.xlabel('mic name')
        combined = np.concatenate((refDelay1, refDelay2))
        min_value = np.min(combined)
        max_value = np.max(combined)
        #plt.yticks(range(min_value-1,max_value+1))
        plt.xticks(range(len(mic_names)), mic_names, fontsize=10)
        plt.grid(True)
        plt.ylabel('delay/samples')
        plt.legend(['near-field','far-field'])
        
        # subplot #2, showing differences
        plt.subplot(212)

        diff = refDelay1-refDelay2
        plt.plot(mic_names,diff,'+-')
        plt.title('talkbox pos # %d' %(talkbox_loc_name))
        
        #plt.yticks(range(int(np.min(diff)),int(np.max(diff))))
        plt.xticks(range(len(mic_names)), mic_names, fontsize=10)
        plt.grid(True)
        plt.ylabel('delay/samples')
        plt.legend(['near-far'])

        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()

        isOutputDirExist = os.path.exists("output")
        if not isOutputDirExist:
            os.makedirs("output")

        plt.savefig('output/delay_error_%s.png' %(talkbox_loc_name))


if __name__ == '__main__':

    # Add[let users to select a pair of log files],Brian,02 Mar 2024
    cameraLogFile = selectLogFile('log','camera')
    fpgaLogFile   = selectLogFile('log','fpga')

    
    df_cam, df_dcam = logToDf(cameraLogFile)
    df_fpga         = fpgaLogToDf(fpgaLogFile)

    # process df_fpga 
    df_fpga_indexed  = df_fpga.groupby('packetIndex')
    micIndices       = df_fpga_indexed['index'].mean()
    start_timestamps = df_fpga_indexed['Timestamp'].first()
    end_timestamps   = df_fpga_indexed['Timestamp'].last()
    average_delays   = df_fpga_indexed['data'].apply(lambda x: x.iloc[1:].head(7).mean())

    # get a summary from df_fpga that includes start and end timestamps, average delay
    df_fpga_summary ={}
    df_fpga_summary['mic index']       = micIndices.values
    df_fpga_summary['start_timestamp'] = start_timestamps.values
    df_fpga_summary['end_timestamp']   = end_timestamps.values
    df_fpga_summary['average_delay']   = average_delays.values
    
    
    # group by talkbox location index and 
    # then get mean values of various colums from df_cam and df_dcam
    df_cam_indexed   = df_cam.groupby('index')
    talkbox_locs     = [v for v in df_cam_indexed.groups.keys()]
    start_timestamps = df_cam_indexed['Timestamp'].first()
    end_timestamps   = df_cam_indexed['Timestamp'].last()

    # get average 3d coordinates from df_cam
    columns_involved =  ["xv", "yv", "zv", "xi", "yi", "roll", "pitch", "yaw"]
    df_cam_vecs      = df_cam_indexed[columns_involved].mean()

    # get average 3d coordinates from df_dcam
    columns_involved =  ['xd','yd','zd']
    df_dcam_vecs     = df_dcam.groupby('index')[columns_involved].mean()


    # get a summary from df_cam, df_dcam that includes start and end timestamps, average 3d coordinates
    df_cam_summary = {}
    df_cam_summary['index']=start_timestamps.index
    df_cam_summary['start_timestamp']=start_timestamps.values
    df_cam_summary['end_timestamp']=end_timestamps.values
    df_cam_summary['average_loc']=df_cam_vecs.values
    df_cam_summary['average_loc2']=df_dcam_vecs.values


      
    # get delays for each of the talkbox indices

    # create "delay.csv" to dump data
    f = open('delays.csv','w')
    f.close()

    ## Jason - 5 Mar 2024
    # delay_phase,raw_delay,sorted_micNames = delay_calculation([0,0,3.7])
    # print("raw_delay: ", raw_delay)

    ## plot delay for near-field, far-field, delay

    plotDelay(talkbox_locs, df_cam_summary, df_fpga_summary)
    # plotDelayFar(talkbox_locs, df_cam_summary, df_fpga_summary)
    #plotTimeDelayError(talkbox_locs, df_cam_summary, df_fpga_summary)






    # plt.figure()
    # plt.plot(-(df_cam_vecs[['xv','yv']].values[:,0]),-(df_cam_vecs[['xv','yv']].values[:,1]),'*')
    # plt.plot(-df_dcam_vecs[['xd','yd']].values[:,0],df_dcam_vecs[['xd','yd']].values[:,1],'*')
    

    # # print(df_cam_vecs,df_dcam_vecs)

    # # print("cam: \n", df_cam[:30])
    # # print("dcam: \n", df_dcam[:30])



    # get ref delays based upon the 
    # df_cam_summary['average_loc'] and
    # df_cam_summary['average_loc2']    
    plt.figure()
    plt.plot(df_cam_summary['average_loc'][:,0]*100,-df_cam_summary['average_loc'][:,1]*100,'*')
    plt.title('talkbox locations')
    plt.xlabel('x/cm')
    plt.ylabel('y/cm')
    plt.savefig('output/talkboxLocs.png')
        
    plt.show()

    # # compare the above two

