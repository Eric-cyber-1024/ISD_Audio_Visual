# simple script to simulate the mic array
#

import matplotlib.pylab as plt
import numpy as np ###### require install and adjust to certain edition 1.13.3
from test_delay_cal import *

SAMPLING_RATE = 48e3
f0 = 1e3

vec = np.array([0.3,0.2,0.87])


_,refDelays,_ = delay_calculation(vec) 

plt.figure()

# scan doa
magnitudes=[]
xrange = np.arange(-3,3,0.1)
for x0 in xrange:
    
    vec[0]=x0
    
    _,delays,_ = delay_calculation(vec)   
    # refDelay = refDelay*48e3
    # refDelay = np.max(refDelay)-refDelay
    # refDelay = np.round(refDelay)

    # plt.figure()
    # plt.plot(refDelay*SAMPLING_RATE)

    # plt.figure()
    time = np.arange(0,100) 
    x= np.zeros_like(time)
    delays  = np.round(delays*SAMPLING_RATE)
    for i, delay in enumerate(delays):
        delayed_time = time + (delay-refDelays[i]*SAMPLING_RATE)
        phase_shift = 2 * np.pi * f0 * delayed_time/SAMPLING_RATE
        x = x + np.sin(phase_shift)
        #plt.plot(delayed_time,np.sin(phase_shift))
        
    x = x/32.
    #print(np.max(x),np.min(x))
    
    magnitude = np.max(abs(x))
    magnitudes.append(magnitude)
    


plt.plot(xrange,magnitudes,'b*-')
plt.grid(True)
plt.xlabel('doa x-range')
plt.ylabel('magnitude')
plt.title('Simulaiton Results')
plt.show()