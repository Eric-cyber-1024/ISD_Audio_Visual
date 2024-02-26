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

    # Load previously saved data
    #with np.load('B.npz') as X:
    #    mtx, dist, _, _ = [X[i] for i in ('arr_0.npy', 'arr_1.npy', 'arr_2.npy', 'arr_3.npy')]
    mtx,dist = loadCalibrationData('20240226_1255_1m6_calibration_err0_05.yaml')

    print(mtx)
    print(dist)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    axisCube = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                           [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

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
        
        if ret == True:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Find the rotation and translation vectors
            ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

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

    cv.destroyWindow()