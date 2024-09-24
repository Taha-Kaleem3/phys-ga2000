import numpy as np
import timeit as timeit
import matplotlib.pyplot as plt


def matrixMultExplicit(N):
    """
    Explicitly multiplies 2 random matrices a and b with dimension N using the method in Newman

    Inputs: N, type: integer. Desc: dimension of 2 random matrices

    Outputs: c, type: np.array(shape= (N, N)). Desc: dot product of 2 random matrices
    """
    a = np.random.rand(N, N)
    b = np.random.rand(N, N)
    c = np.zeros([N, N], np.float32())
    for i in range(N):
        for j in range(N):
            for k in range(N):
                c[i, j]+=a[i, k] + b[k, j]
    return c


def matrixMultDot(N):
    """
    <ultiplies 2 random matrices a and b with dimension N using numpy's dot function

    Inputs: N, type: integer. Desc: dimension of 2 random matrices

    Outputs: c, type: np.array(shape= (N, N)). Desc: dot product of 2 random matrices
    """
    a = np.random.rand(N, N)
    b = np.random.rand(N, N)
    c = np.dot(a, b)
    return c

# Defining the maximum dimension for the multiplication, creating a list of dimensions to plot on the x axis
# and creating 2 arrays with the time in takes the matrix multiplication to take place
N = 100
dimensions = np.arange(start = 0, stop = N, dtype=int)
multTimeDot = np.zeros(len(dimensions))
multTimeExpl = np.zeros(len(dimensions))

N3 = dimensions ** 3

# CDot = []
# CExpl = []

for n in dimensions:
    multTimeExpl[n] = timeit.timeit(lambda: matrixMultExplicit(n), number = 1)
    multTimeDot[n] = timeit.timeit(lambda: matrixMultDot(n), number = 1)
    # if (n != 0):
    #     CDot.append(multTimeDot[n]/ (n** 3))
    #     CExpl.append(multTimeExpl[n]/ (n ** 3))
    # else:
    #     CDot.append(0)
    #     CExpl.append(0)

fig, axes = plt.subplots(1, 2, figsize = (16, 8))

# CDotMean = np.mean(np.array(CDot))
# CExplMean = np.mean(np.array(CExpl))

axes[0].scatter(dimensions, multTimeDot)
axes[1].scatter(dimensions, multTimeExpl)


# axes[0].plot(dimensions, CDotMean * N3, label = "N^3")
# axes[1].plot(dimensions, CExplMean * N3, label = "N^3")

axes[0].set_title("Dot function multiplication")
axes[1].set_title("Explicit function multiplication")

axes[0].set_xlabel("Dimension of matrixes")
axes[0].set_ylabel("Computation time (s)")

axes[1].set_xlabel("Dimension of matrixes")
axes[1].set_ylabel("Computation time (s)")

plt.savefig("ps-3/plots/matrixMultiplication")