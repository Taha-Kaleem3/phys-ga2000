import numpy as np


smallestvalue32 = np.float32(	1 * 10**-7)
print("approximately smallest sum of 1 of float 32:" + str(smallestvalue32+np.float32(1.0)))

smallestvalue64 = np.float64(	1 * 10**-15)
print("approximately smallest sum of 1 of float 64:" + str(smallestvalue64+np.float64(1.0)))

minvaluesum32 = np.float32( 1 * 10**-7)
print("... minimum positive float32 without underflow: " + str(minvaluesum32-np.float32(1.0)))

minvaluesum64 = np.float64(1* 10**-16)
print("... minimum positive float64 without underflow: " + str(minvaluesum64-np.float64(1.0)))

maxvaluesum32 = np.float32( 9.999999999999 * 10**37)
print("... maximum positive float32 without overflow: " + str(maxvaluesum32+np.float32(1.0)))

maxvaluesum64 = np.float64(9.9999999999999* 10**307)
print("... maximum positive float64 without overflow: " + str(maxvaluesum64+np.float64(1.0)))