import numpy as np
import matplotlib.pyplot as plt

N = 10000

def Mandelbrot(c, it):
    z = c
    for i in range(100):
        z = (z ** 2) + c
        tf_array = np.abs(z <= 2)
        it[tf_array] = i

    it[np.abs(z == 0)] = 100
    return it

x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)
X, Y = np.meshgrid(x, y)

c = X + Y*1j

i = np.zeros(len(X) ** 2).reshape(c.shape)
iterations = Mandelbrot(c, i)



plt.imshow(iterations, extent=(-2, 2, -2, 2), cmap="autumn")
plt.colorbar(label='intensity')


plt.xlabel('real')
plt.ylabel('imaginary')
plt.title('Mandelbrot Set')
plt.savefig("PS-2/plots/mandelbrot")

