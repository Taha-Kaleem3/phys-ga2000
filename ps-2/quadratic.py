import numpy as np
def quadraticNormal(a, b, c):
    return np.float32((-b+np.sqrt(b ** 2 - 4 * a *c))/(2*a)), np.float32((-b-np.sqrt(b ** 2 - 4 * a *c))/(2*a))

def quadraticInverted(a, b, c):
    return np.float32((2*c)/(-b-np.sqrt(b **2 -4*a*c))), np.float32((2*c)/(-b+np.sqrt(b **2 -4*a*c)))
print(quadraticNormal(0.001, 1000, 0.001))
print(quadraticInverted(0.001, 1000, 0.001))

def quadratic(a, b, c):
    if(b>=0):
        return np.float32((2*c)/(-b-np.sqrt(b **2 -4*a*c))), np.float32((-b-np.sqrt(b ** 2 - 4 * a *c))/(2*a))
    if (b<0):
        return np.float32((-b+np.sqrt(b ** 2 - 4 * a *c))/(2*a)), np.float32((2*c)/(-b+np.sqrt(b **2 -4*a*c)))
print(quadratic(0.001, 1000, 0.001))
