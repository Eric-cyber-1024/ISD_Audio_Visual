# simple python script to calibrate camera
# ref: 
#      https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html
#      https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
#      https://learnopencv.com/camera-calibration-using-opencv/
#
# revised from: https://github.com/zeryabmoussaoui/camera_calibration_tool.git


# Author : Zeryab Moussaoui (zeryab.moussaoui@gmail.com)
# Description : Compute calibration matrix from (web)camera
# Inspired from : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# Version : 1.0

from pywinauto import Desktop
import numpy as np
import cv2
import yaml

# Parameters
#TODO : Read from file

# the followings should match with the calibration checkerboard rig
n_row=6 
n_col=9
n_min_img = 10 # img needed for calibration
square_size=38 # checkboard square size in mm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # termination criteria
corner_accuracy = (11,11)
result_file = "./calibration.yaml"

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(n_row-1,n_col-1,0)
objp = np.zeros((n_row*n_col,3), np.float32)
objp[:,:2] = np.mgrid[0:n_row,0:n_col].T.reshape(-1,2)*square_size



# Intialize camera and window
camera = cv2.VideoCapture(1) #Supposed to be the only camera
if not camera.isOpened():
    print("Camera not found!")
    quit()
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.namedWindow("Calibration",cv2.WINDOW_NORMAL)
cv2.resizeWindow('Calibration',width,height)


def loadCalibrationData(yamlFileName):
    with open(yamlFileName, 'r') as file:
        data = yaml.safe_load(file)

    # Extract the camera matrix and distortion coefficients
    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['dist_coeff'])
    print(camera_matrix,dist_coeffs)
    return camera_matrix,dist_coeffs


# draw a 3D cube wireframe for checking calibration results
def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img


# Usage
def usage():
    print("Press on displayed window : \n")
    print("[space]     : take picture")
    print("[c]         : compute calibration")
    print("[r]         : reset program")
    print("[l]         : load cal data")
    print("[ESC]    : quit")

usage()
Initialization = True

while True:    
    if Initialization:
        print("Initialize data structures ..")
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        n_img = 0
        Initialization = False
        tot_error=0
    
    # Read from camera and display on windows
    ret, img = camera.read()
    cv2.imshow("Calibration", img)
    if not ret:
        print("Cannot read camera frame, exit from program!")
        camera.release()        
        cv2.destroyAllWindows()
        break
    
    # Wait for instruction 
    k = cv2.waitKey(50) 
   
    # SPACE pressed to take picture
    if k%256 == 32:   
        print("Adding image for calibration...")
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(imgGray, (n_row,n_col),None)

        # If found, add object points, image points (after refining them)
        if not ret:
            print("Cannot found Chessboard corners!")
            
        else:
            print("Chessboard corners successfully found.")
            objpoints.append(objp)
            n_img +=1
            corners2 = cv2.cornerSubPix(imgGray,corners,corner_accuracy,(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            imgAugmnt = cv2.drawChessboardCorners(img, (n_row,n_col), corners2,ret)
            cv2.imshow('Calibration',imgAugmnt) 
            cv2.waitKey(500)  

            print('number of object points %d' %(len(objpoints)))      
                
    # "c" pressed to compute calibration        
    elif k%256 == 99:        
        if n_img <= n_min_img:
            print("Only ", n_img , " captured, ",  " at least ", n_min_img , " images are needed")
        
        else:
            print("Computing calibration ...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width,height),None,None)

            

            
            if not ret:
                print("Cannot compute calibration!")
            
            else:
                print("Camera calibration successfully computed")
                # Compute reprojection errors
                for i in range(len(objpoints)):
                   imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                   error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                   tot_error += error
                print("Camera matrix: ", mtx)
                print("Distortion coeffs: ", dist)
                print("Total error: ", tot_error)
                print("Mean error: ", np.mean(error))
                
                # Saving calibration matrix

                rvecs = np.array(list(rvecs))
                tvecs = np.array(list(tvecs))

                print('number of cal points: %d' %(len(objpoints)))
                print(rvecs.shape,tvecs.shape)

                print("Saving camera matrix .. in ",result_file)
                data={"camera_matrix": mtx.tolist(), "dist_coeff": dist.tolist(),"rvecs":rvecs.tolist(),"tvecs":tvecs.tolist()}
                with open(result_file, "w") as f:
                    yaml.dump(data, f, default_flow_style=False)

                # project 3D points to image plane
                # axes = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
                # imgPtsAll=[]
                # for i in range(len(axes)):
                #     imgpts, jac = cv2.projectPoints(axes[i], rvecs[i], tvecs[i], mtx, dist)
                #     imgPtsAll.append(imgpts)
                # img = drawCube(img,corners2,imgPtsAll)
                # cv2.imshow('Calibration',img) 
                # cv2.waitKey(1000)   
                
    # ESC pressed to quit
    elif k%256 == 27:
            print("Escape hit, closing...")
            camera.release()        
            cv2.destroyAllWindows()
            break
    # "r" pressed to reset
    elif k%256 ==114: 
         print("Reset program...")
         Initialization = True
    
    # 'l' pressed to load yaml calibration data
    elif k%256 == ord('l'):
        print('load calibration data from yaml file')
        loadCalibrationData('calibration.yaml')
        
            
        
        
        
        