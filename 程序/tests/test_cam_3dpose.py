# ref:
# https://docs.opencv.org/3.4/d7/d53/tutorial_py_pose.html
# https://github.com/wuxinwang1997/OpenCV-Python-Tutorials/blob/master/Camera%20Calibration%20and%203D%20Reconstruction/Pose%20Estimation/PoseEstimation.py

from pywinauto import Desktop  # add this to handle UI scaling issue
import cv2 as cv
import numpy as np
import glob
import yaml


def loadCalibrationData(yamlFileName):
    with open(yamlFileName, 'r') as file:
        data = yaml.safe_load(file)

    # Extract the camera matrix and distortion coefficients
    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['dist_coeff'])
    #print(camera_matrix,dist_coeffs)
    return camera_matrix,dist_coeffs

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    
    img = cv.line(img, (int(corner[0]),int(corner[1])), (int(imgpts[0][0][0]),int(imgpts[0][0][1])), (255, 0, 0), 5)
    img = cv.line(img, (int(corner[0]),int(corner[1])), (int(imgpts[1][0][0]),int(imgpts[1][0][1])), (0, 255, 0), 5)
    img = cv.line(img, (int(corner[0]),int(corner[1])), (int(imgpts[2][0][0]),int(imgpts[2][0][1])), (0, 0, 255), 5)
    return img


def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground flooe in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


if __name__ == '__main__':

    n_row=6 
    n_col=9
    square_size = 38 # 38 mm 

    # Define the font settings
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.
    font_color = (255, 255, 255)  # White color
    line_type = cv.LINE_AA

    # Write the debug message on the image
    debug_message = "Debug Message"
    text_size, _ = cv.getTextSize(debug_message, font, font_scale, 1)
    text_x = 10
    text_y = 10 + text_size[1]


    # Load previously saved data
    #with np.load('B.npz') as X:
    #    mtx, dist, _, _ = [X[i] for i in ('arr_0.npy', 'arr_1.npy', 'arr_2.npy', 'arr_3.npy')]
    mtx,dist = loadCalibrationData('20240226_1255_1m6_calibration_err0_05.yaml')

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # 3D object points, assuming the chessboard lies on the X-Y plane (z=0) and square size of 38mm
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)*square_size

   

    axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)*square_size
    axisCube = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                           [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])*square_size

    camera = cv.VideoCapture(1) #Supposed to be the only camera
        
    #for fname in glob.glob('left*.jpg'):
    while True:
        #img = cv.imread(fname)
        ret, img = camera.read()
        if not ret:
            print("Cannot read camera frame, exit from program!")
            camera.release()        
            cv.destroyAllWindows()
            break
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (6, 9), None)
        
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            imgAugmnt = cv.drawChessboardCorners(img, (n_row,n_col), corners2,ret)
            cv.imshow('Pose Estimation',imgAugmnt) 

            # Find the rotation and translation vectors
            # assuming word coordinates lies on the chessboard 
            ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

            if ret:
                debug_message='%.2f,%.2f,%.2f' % (tvecs[0]/1e3,tvecs[1]/1e3,tvecs[2]/1e3)
                cv.putText(img, debug_message, (text_x, text_y), font, font_scale, font_color, 1, line_type)
                
                # converting Rodrigues format to 3x3 rotation matrix format
                rotMatrix,_=cv.Rodrigues(rvecs)
                #print(rotMatrix)
            

                # project 3D points to image plane
                imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

                img = draw(img, corners2, imgpts)
                cv.imshow('Pose Estimation', img)
                imgptsCube, jacCube = cv.projectPoints(axisCube, rvecs, tvecs, mtx, dist)
                imgCube = drawCube(img, corners2, imgptsCube)
                cv.imshow('Pose Estimation Cube', imgCube)
            
            
            k = cv.waitKey(50) & 0xff
            if k == 'c':
                continue
            elif k%256 == 27:
                print("Escape hit, closing...")
                camera.release()        
                break

        else:
            cv.imshow('Pose Estimation', img)
            k = cv.waitKey(50) & 0xff
            if k%256 == 27:
                print("Escape hit, closing...")
                camera.release()        
                break

    cv.destroyAllWindows()