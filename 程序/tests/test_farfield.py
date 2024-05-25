import numpy as np
import matplotlib.pylab as plt

def getDistance(rs,rm):
    '''
    
    return distance | rs-rm|

    '''
    d = np.linalg.norm((rs - rm),axis=0) - np.linalg.norm(rs, axis=0)

    return d


def getDistance_farfield(rs,rm):
    '''
    get distance based upon far-field approximation

    return rm dot unit vector of rs

    '''
    srcLocsNorms= np.linalg.norm(srcLocs,axis=0,keepdims=True)
    unitVecs = srcLocs/srcLocsNorms


    # Calculate the dot product of each column
    d = np.sum(micLocs*unitVecs, axis=0)

    return d


def generateTestVectors(N,K1,K2,z1=5.,z2=8.):
    '''
    generate random test vectors 

    '''
    

    # generate micLocs on the xy plane (z=0)
    micLocs = np.random.uniform(-K1, K1, size=(2, N))
    micLocs = np.vstack((micLocs, np.zeros((1, N))))

    # generate ref_micLocs on the xy plane (z=0)
    ref_micLocs = np.random.uniform(-K1, K1, size=(2, N))
    ref_micLocs = np.vstack((ref_micLocs, np.zeros((1, N))))

    # generate srcLocs z ranges from 3m to 8m
    srcLocs = np.random.uniform(-K2, K2, size=(2, N))
    srcLocs = np.vstack((srcLocs, np.random.uniform(z1,z2,size=(1,N))))

    return srcLocs,micLocs,ref_micLocs


def loadFixedTestVectors(N,K2,z1=5.,z2=8.):
    '''
    
    generate test vectors out of a small subset of vectors


    - mic name: "M01"
    location: [0.2, 0.15, 0.0]

    - mic name: "M09"
    location: [0.4,-0.12, 0.0]

    - mic name: "M17"
    location: [-0.2, 0.15, 0.0]

    - mic name: "M25"
    location: [-0.4,-0.12, 0.0]


    '''

    # Create four example arrays of shape (1, 3)
    m01 = np.array([[0.2, 0.15, 0.0]])
    m09 = np.array([[0.4,-0.12, 0.0]])
    m17 = np.array([[-0.2, 0.15, 0.0]])
    m25 = np.array([[-0.4,-0.12, 0.0]])

    # List of arrays
    arrays = [m09,m17,m25]


    # Randomly select N column indices from the range of available columns
    column_indices = np.random.choice(3, size=N)

    # Initialize the micLocs matrix
    micLocs = np.zeros((3, N))

    # Populate the output matrix with randomly selected columns
    for i, col_idx in enumerate(column_indices):
        micLocs[:, i] = arrays[col_idx][0]
        

    # generate the ref_micLocs 
    ref_micLocs = np.repeat(m01,N,axis=0).T


    # generate srcLocs z ranges from 3m to 8m
    srcLocs = np.random.uniform(-K2, K2, size=(2, N))
    srcLocs = np.vstack((srcLocs, np.random.uniform(z1,z2,size=(1,N))))

    return srcLocs,micLocs,ref_micLocs



if __name__ == '__main__':

    N = 500000  # Number of vectors
    
    # generate test vectors 
    K1 = 0.25
    K2 = 0.01
    z1 = 5.
    z2 = 8.
    # srcLocs, micLocs, ref_micLocs = generateTestVectors(N,K1,K2,z1,z2)

    srcLocs, micLocs, ref_micLocs = loadFixedTestVectors(N,K2,z1,z2)

    # calculate d1 = | rs - rm | - | rs | 

    d1 = getDistance(srcLocs,micLocs)
    d2 = getDistance(srcLocs,ref_micLocs)
    d3 = d1-d2

    d1_farfield = getDistance_farfield(srcLocs,micLocs)
    d2_farfield = getDistance_farfield(srcLocs,ref_micLocs)
    d3_farfield = getDistance_farfield(srcLocs,micLocs-ref_micLocs)

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(micLocs[0,:],micLocs[1,:],'*',ref_micLocs[0,:],ref_micLocs[1,:],'r*')
    # plt.subplot(212)
    # plt.plot(srcLocs[0,:],srcLocs[2,:],'*')
    # plt.figure()
    # #plt.plot(srcLocs[2,:],np.abs(d1-d1_farfield)*1e6/340,'*')
    # #plt.plot(srcLocs[2,:],np.abs(d2-d2_farfield)*1e6/340,'r*')
    # plt.plot(srcLocs[2,:],np.abs(d3-d3_farfield)*1e6/340,'*')
    # plt.xlabel('|rs|/m')
    # plt.ylabel('time delay/us')
    plt.figure()
    plt.hist(np.abs(d3-d3_farfield)*1e6/340,100)
    plt.show()