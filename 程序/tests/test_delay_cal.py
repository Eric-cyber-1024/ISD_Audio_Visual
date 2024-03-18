import numpy as np
import matplotlib.pyplot as plt
import yaml
import math

NUM_OF_MICS    = 32
SPEED_OF_SOUND = 340.
OUTER_RADIUS   = 0.25
MID_RADIUS     = 0.20
INNER_RADIUS   = 0.15

MIC_NAMES      = np.array([
'M01',
'M03',
'M06',
'M08',
'M17',
'M18',
'M21',
'M24',
'M25',
'M29',
'M31',
'M32',
'M09',
'M10',
'M11',
'M13',
'M02',
'M05',
'M19',
'M22',
'M26',
'M27',
'M16',
'M15',
'M04',
'M07',
'M20',
'M23',
'M28',
'M30',
'M12',
'M14',
'M00'
],dtype='U3')


def getMicPositions(xOffset,yOffset,zOffset):
    '''
    
    get 3d coordinates of the microphones of the microphone array

    '''
    angles= np.arange(0,2*np.pi,np.pi/8)
    angles2=np.arange(0,2*np.pi,np.pi/4)

    outer = OUTER_RADIUS*np.array([np.sin(angles),np.cos(angles)])
    mid   = MID_RADIUS*np.array([np.sin(angles2+np.pi/8),np.cos(angles2+np.pi/8)])
    inner = INNER_RADIUS*np.array([np.sin(angles2),np.cos(angles2)])
    ref   = np.array([[0],[0]]) # at the center of the microphone array, a virtual one

    overall = np.concatenate((outer,inner,mid,ref),axis=1)
    overall = np.concatenate([overall,np.zeros((1,NUM_OF_MICS+1))],axis=0)

    overall[0,:]+=xOffset
    overall[1,:]+=yOffset
    overall[2,:]+=zOffset
    
    sorted_indices = np.argsort(MIC_NAMES)
    overall = overall[:,sorted_indices]
    return overall.T

def delay_calculation(src_position,xOffset,yOffset,zOffset):
    '''
    Calculates the delay phase for a given source position based on the microphone positions.

    Parameters:
        src_position (list or array): The source position in the format [x, y].

    Returns:
        tuple: A tuple containing the following:
            - delay_phase (ndarray): The delay phase for each microphone.
            - raw_delay (ndarray): The raw delay values for each microphone.
            - sorted_micNames (ndarray): The sorted microphone names.

    '''

    src_position = np.array(src_position)  # this_location=[6, actual.x, acutal.y]
    
    
    mic_position = getMicPositions(xOffset,yOffset,zOffset)  # getting mic position

    sorted_micNames = np.sort(MIC_NAMES)
    
    mic_ref_ori = mic_position[sorted_micNames=='M01'][0]    

    

    # vector difference between ref mic and the source
    source_ref = src_position - mic_ref_ori

    # vector difference between ref mic and the other mics
    ref_mics = np.zeros((NUM_OF_MICS))
    # for i in range(NUM_OF_MICS):
    #     ref_mics[i] = mic_ref_ori - mic_position[i] 
    
    
    # distances of mics vs source
    magnitude_s2p = np.zeros((NUM_OF_MICS))

    # distance between ref mic and source
    magnitude_s2r = np.linalg.norm(source_ref, axis=0, keepdims=True)
 

    # for each of the mics
    for i in range(NUM_OF_MICS):
        # get the distance between mics and source
        vec_diff = src_position - mic_position[i] # get the vector difference first
        magnitude_s2p[i] = np.linalg.norm(vec_diff, axis=0, keepdims=True)
        # print(magnitude_s2p[i]*1e6/SPEED_OF_SOUND)

    delay = np.zeros((NUM_OF_MICS))
    for i in range(NUM_OF_MICS):
        delay[i] = - (magnitude_s2r - magnitude_s2p[i]) / SPEED_OF_SOUND

    # save a copy of the raw delays
    raw_delay = delay

    # delay adjustments to get the delays need to add to match all the mic signals with the one with max. delay
    delay = max(delay)-delay
    delay_phase = delay*48000.  # 48KHz
    delay = (np.round(delay / 1e-6, 6))


    
    # mic_delay_extra = find_calibration_value(Vision.distance,Vision.x,Vision.y)
    for i in range(NUM_OF_MICS):
        delay[i] = (delay[i])*1e-6
        delay_phase[i] = delay[i]*48000.
        delay_phase[i] = int(256*delay_phase[i] *(360./512.))
            
    minimum = abs(min(delay_phase))
    delay_phase = delay_phase + minimum
    delay_phase = np.reshape(delay_phase, NUM_OF_MICS)

    return delay_phase,raw_delay,sorted_micNames 


# Jason - 5 Mar 2024
def delay_calculation_far(vec_uv):
    
    # load calibration data
    k,d = loadCalibrationData('20240226_1255_1m6_calibration_err0_05.yaml')
    fx, zero, cx, zero, fy, cy, zero, zero, one = list(np.concatenate(k).flat)  

    sorted_micNames = np.sort(MIC_NAMES)
    mic_position = getMicPositions(0,0.5,0)  # getting mic position
    mic_ref_ori = mic_position[sorted_micNames=='M01'][0]

    # vector difference between ref mic and the other mic
    rmk = mic_ref_ori - mic_position
    
    vec_uv_c = [(vec_uv[0]-cx)/fx, (vec_uv[1]-cy)/fy, 1]
    vec_uv_c_normalized = vec_uv_c / np.linalg.norm(vec_uv_c, axis=0)
    # print("np.linalg.norm(vec_uv_c, axis=0): ", np.linalg.norm(vec_uv_c, axis=0))
    # print("vec_uv_c_normalized: ", vec_uv_c_normalized)

    delay = np.zeros((NUM_OF_MICS))
    for i in range(NUM_OF_MICS):
        delay[i] = np.matmul(rmk[i], np.reshape(vec_uv_c, (-1, 1)))

    delay = delay / SPEED_OF_SOUND / np.linalg.norm(vec_uv_c, axis=0)
    print("delay: ", delay)
    return vec_uv_c_normalized, delay, sorted_micNames


def delay_calculation_eq2(srcCoordinatesW):

    src_position = srcCoordinatesW
    mic_position = getMicPositions(0,0.5,0)  # getting mic position
    sorted_micNames = np.sort(MIC_NAMES)
    mic_ref_ori = mic_position[sorted_micNames=='M01'][0]  

    # vector difference between ref mic and the other mics
    ref_mics = np.zeros((NUM_OF_MICS,3))
    for i in range(NUM_OF_MICS):
        ref_mics[i] = mic_ref_ori - mic_position[i] 

    src_mag = math.sqrt(src_position[0]**2 + src_position[1]**2 + src_position[2]**2)
    rs_not = np.array([src_position[0]/src_mag, src_position[1]/src_mag, src_position[2]/src_mag])

    delay = np.zeros((NUM_OF_MICS))

    for i in range(NUM_OF_MICS):
        delay[i] = np.matmul(ref_mics[i], np.reshape(rs_not, (-1,1))) / SPEED_OF_SOUND

    return delay, sorted_micNames



def loadCalibrationData(yamlFileName):
    with open(yamlFileName, 'r') as file:
        data = yaml.safe_load(file)

    # Extract the camera matrix and distortion coefficients
    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['dist_coeff'])
    #print(camera_matrix,dist_coeffs)
    return camera_matrix,dist_coeffs


if __name__ == '__main__':

    delay_phase,raw_delay,sorted_micNames = delay_calculation([0,0,3.7])


