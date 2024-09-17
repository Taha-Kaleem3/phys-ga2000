import numpy as np
import timeit as timeit
e = 1.602176634 * 10**-19 #Electric charge in coulombs
a = 5.63 * 10 ** -10 #distance between sodium chloride atoms (m)
e0 = 8.85418782 * 10 ** -12 # permittivity of free space (m^-3 kg^-1 s^4 A^2)

"""
Without for loop
"""
def Vijk(i,j,k):
    if(i == 0 and j == 0 and k == 0):
        return 0
    sign = (-1) ** ((i+j+k)+1)
    val = e/(4*np.pi*e0*a*np.sqrt(i**2+j**2+k**2))
    return sign * val

def VijkMask(i,j,k, zeroMask):
    i = i[zeroMask]
    j = j[zeroMask] 
    k = k[zeroMask]
    sign = (-1) ** ((i+j+k)+1)
    val = e/(4*np.pi*e0*a*np.sqrt(i**2+j**2+k**2))
    return sign * val

def Vtot(L):
    # matrixFunVijk = np.vectorize(Vijk)
    ijkcoords = np.linspace(-L, L, 2*L+1)
    i, j, k = np.meshgrid(ijkcoords, ijkcoords, ijkcoords)
    zeroMask =  (i != 0) |  (j != 0) |  (k != 0)


    result = VijkMask(i, j, k, zeroMask)
    sum = np.sum(result)
    return sum

def Madelung(L):
    return Vtot(L)*(4*np.pi*e0*a/e)

n = 200
madelung = Madelung(n)
print(f"Madelung constant: {madelung}")

def codeToTime():
    Madelung(n)
time = timeit.timeit(codeToTime, number = 1)
print(f"time: {time} s")

"""
With for loop
"""
def MadelungFor(L):
    sum = 0
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                sum += Vijk(i, j, k)
    return sum*(4*np.pi*e0*a/e)

madelung = MadelungFor(n)
print(f"Madelung constant (for loop): {madelung}")

def codeToTimeFor():
    MadelungFor(n)
time2 = timeit.timeit(codeToTimeFor, number = 1)
print(f"time (for loop): {time2} s")
