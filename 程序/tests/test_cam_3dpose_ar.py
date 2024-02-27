'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100

Dependency:
opencv-contrib-python==4.7.0.72

'''

from pywinauto import Desktop  # add this to handle UI scaling issue
import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import yaml


def draw(img, corners, imgpts):

    # why it has to be corners[1]?!
    corner = tuple(corners[1].ravel())
    
    # blue -- X
    img = cv2.line(img, (int(corner[0]),int(corner[1])), (int(imgpts[0][0][0]),int(imgpts[0][0][1])), (255, 0, 0), 5)
    # green -- Y 
    img = cv2.line(img, (int(corner[0]),int(corner[1])), (int(imgpts[1][0][0]),int(imgpts[1][0][1])), (0, 255, 0), 5)
    # red -- Z
    img = cv2.line(img, (int(corner[0]),int(corner[1])), (int(imgpts[2][0][0]),int(imgpts[2][0][1])), (0, 0, 255), 5)
    return img


def loadCalibrationData(yamlFileName):
    with open(yamlFileName, 'r') as file:
        data = yaml.safe_load(file)

    # Extract the camera matrix and distortion coefficients
    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['dist_coeff'])
    #print(camera_matrix,dist_coeffs)
    return camera_matrix,dist_coeffs

def pose_esitmation(frame, aruco_dict_type, mtx, dist):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    # Define the font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.
    font_color = (255, 255, 255)  # White color
    line_type = cv2.LINE_AA

    # Write the debug message on the image
    debug_message = "Debug Message"
    text_size, _ = cv2.getTextSize(debug_message, font, font_scale, 1)
    text_x = 10
    text_y = 10 + text_size[1]
    background_color = (50, 50, 50)  # dark grey background color

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    #parameters = cv2.aruco.DetectorParameters_create()

    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    # 3D object points, assuming the chessboard lies on the X-Y plane (z=0) and square size of 38mm
    square_size=38
    objp = np.zeros((4, 3), np.float32)
    #objp[:, :2] = np.mgrid[0:2, 0:2].T.reshape(-1, 2)*square_size
    objp[0]=[0.,38.,0.]
    objp[1]=[0.,0.,0.]
    objp[2]=[38.,0.,0.]
    objp[3]=[38.,38.,0.]

    axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)*square_size


    corners, ids, rejected_img_points = detector.detectMarkers(gray)

    

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            if ids[i][0]==590:
                
                
                # # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                # rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                #                                                            distortion_coefficients)
                # # Draw a square around the markers
                #cv2.aruco.drawDetectedMarkers(frame, [corners[i]]) 

                # # Draw Axis
                # cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  

                cc = corners[i][0]
                cc = np.reshape(cc, (cc.shape[0], 1, 2))
                ret, rvecs, tvecs = cv2.solvePnP(objp,cc, mtx, dist)

                if ret:
                    
                    debug_message='%.2f,%.2f,%.2f' % (tvecs[0]/1e3,tvecs[1]/1e3,tvecs[2]/1e3)

                    # Get the text size
                    (text_width, text_height), _ = cv2.getTextSize(debug_message, font, font_scale, 1)

                    # Calculate the background rectangle coordinates
                    background_rect_coords = ((0, 0), (text_width + 10, text_height + 10))  # Adjust padding as needed

                    # Draw the background rectangle
                    cv2.rectangle(frame, background_rect_coords[0], background_rect_coords[1], background_color, cv2.FILLED)
                    cv2.putText(frame, debug_message, (text_x, text_y), font, font_scale, font_color, 1, line_type)
                
                    
                    # converting Rodrigues format to 3x3 rotation matrix format
                    rotMatrix,_=cv2.Rodrigues(rvecs)
                    #print(rotMatrix)
                

                    # project 3D points to image plane
                    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

                    frame = draw(frame, cc, imgpts)
                    #cv2.imshow('Estimated Pose', img)
                    

    return frame

if __name__ == '__main__':

    opencv_version = cv2.__version__

    # make sure that opencv version is okay!!
    assert opencv_version == '4.7.0'

    ap = argparse.ArgumentParser()
    # ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    # ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]

    # calibration_matrix_path = args["K_Matrix"]
    # distortion_coefficients_path = args["D_Coeff"]
    
    # k = np.load(calibration_matrix_path)
    # d = np.load(distortion_coefficients_path)
    
    # load calibration data
    k,d = loadCalibrationData('20240226_1255_1m6_calibration_err0_05.yaml')

    video = cv2.VideoCapture(1)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()
        
        if not ret:
            break
        
        output = pose_esitmation(frame, aruco_dict_type, k, d)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF

        # press q or ESC to quit
        if key == ord('q') or (key%256)==27:
            break

    video.release()
    cv2.destroyAllWindows()