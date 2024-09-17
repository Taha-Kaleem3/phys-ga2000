import numpy as np
import matplotlib.pyplot as plt

N = 200

def Mandelbrot(c, it):
    z = c
    iterations = 0
    for i in range(100):
        z = (z ** 2) + c
        it[np.abs(z>2)] = i

    it[np.abs(z == 0)] = 100
    return it

x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)
X, Y = np.meshgrid(x, y)

c = X + Y*1j
i = np.zeros(len(c))
iterations = Mandelbrot(c, i)
print(iterations)



iterations = iterations.reshape((len(X), 2))


plt.imshow(iterations, extent=(-2, 2, -2, 2))
# plt.colorbar(label='Count')


# plt.xlabel('real')
# plt.ylabel('imaginary')
# plt.title('Mandelbrot Set')
plt.savefig("PS-2/plots/mandelbrot")

