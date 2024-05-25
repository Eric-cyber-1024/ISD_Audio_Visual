'''
simple python script to study data from intel depth camera d435
date: 24 May 2024
by: Brian Leung
company: LSCM

'''
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import scipy

'''
D435i Depth Intrinsics according to resolution configured
[ 1280x720  p[643.715 362.013]  f[637.674 637.674]  Brown Conrady [0 0 0 0 0] ] – 1280x720
[ 848x480  p[426.461 241.333]  f[422.459 422.459]  Brown Conrady [0 0 0 0 0] ] – 848x480
'''


def linearRegress(x,y):
    '''
    perform linear regression on y against x

    '''
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

    return slope, intercept, r_value


if __name__ == "__main__":

    fileName = '20240524_1800_d435_studydata.txt'

    # the depth camera intrinsics
    vec_p = np.array([639.00036621, 363.5682373])
    vec_f = np.array([909.78485107, 909.87225342])

    with open(fileName,'r') as f:
        rows = f.readlines()



    # Create an empty list to store the data
    data_list = []

    # Iterate over the rows and extract the data
    for row in rows:
        cols = row.split()
        
        # Extract the floating-point numbers
        numbers =[float(x) for x in re.findall(r"-?\d+\.?\d*", " ".join(cols))]


        i, j, depthpixel_x, depthpixel_y, x, y, z, depth = numbers
        data_list.append([i, j, depthpixel_x, depthpixel_y, x, y, z, depth])

    # Create the Pandas DataFrame
    df = pd.DataFrame(data_list, columns=['i', 'j', 'depthpixel x', 'depthpixel y', 'x', 'y', 'z', 'depth'])

    # estimate x,y,z using depth intrinsics and depth
    # step 1. prepare the intrinsic matrix from p and f 
    # step 2. prepare the input vector 
    # step 3. compute the matrix vector product to get (xc,yc,zc)
    # step 4. compare against the input x,y,z

    
    # the camera intrinsic matrix
    m = np.zeros([3,3])
    m[0,0]=vec_f[0]
    m[1,1]=vec_f[1]
    m[0,2]=vec_p[0]
    m[1,2]=vec_p[1]
    m[2,2]=1.

    # inverse of the camera intrinsic matrix
    inv_m = np.zeros([3,3])
    inv_m[0,0]=1/vec_f[0]
    inv_m[1,1]=1/vec_f[1]
    inv_m[0,2]=-vec_p[0]/vec_f[0]
    inv_m[1,2]=-vec_p[1]/vec_f[1]
    inv_m[2,2]=1.


    nCols       = len(df['depth'].values)
    depths      = df['depth'].values
    depthPixels = df[['depthpixel x','depthpixel y']].values.T
    depthPixels = np.vstack((depthPixels, np.ones((1, nCols))))



    positions  = inv_m@depthPixels

    estPositions = np.array([positions[:,i]*depths[i] for i in range(nCols)]).T



    plt.figure()
    plt.subplot(211)
    plt.plot(df['z'],df['depth'],'*-')
    plt.xlabel('z/m')
    plt.ylabel('depth/m')
    plt.title('depth vs z')
    plt.subplot(212)
    plt.plot(df['i'],df['j'],'*')
    plt.xlabel('self i')
    plt.ylabel('self j')
    plt.title('self i vs j')


    for i in range(3):
        x = estPositions[i,:]
        if i==0:
            y = df['x']
        elif i==1:
            y = df['y']
        else:
            y = df['z']

        slope, intercept, r_value = linearRegress(x,y)
        print(i,r_value,slope,intercept)

    # plt.figure()
    # # plt.plot(estPositions[0,:],df['x'],'*-')
    # plt.plot(estPositions[1,:],df['y'],'*-')
    # # plt.plot(estPositions[2,:],df['depth'],'*-')

    
    # plt.show()

