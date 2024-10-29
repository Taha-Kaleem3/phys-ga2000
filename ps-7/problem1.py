import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
m_moon = 7.348e22
m_earth = 5.974e24
m_jupiter = 1.898e27
m_sun = 1.9891e30
R_earth2sun = 1.4857e11
R_earth2moon = 3.844e8
G = 6.674e-11

def rescaledLP(rp, mp):
    return 1/(rp**2) - mp/((1-rp) **2) - 2




rgrid = (np.arange(100000) / 100000.)  
def newton_raphson(mp, n, xst=0.1, func = rescaledLP):
    dfunc = jax.grad(func, argnums=0)
    tol = 1.e-7
    begiter = 6000
    maxiter = 7000
    x = xst
    plt.plot(rgrid, rescaledLP(rgrid, mp))
    plt.plot(rgrid, 0. * rescaledLP(rgrid, mp))
    plt.ylim((-3, 3))
    for i in np.float32(np.arange(begiter, maxiter))/(maxiter-begiter):
        if(i != 1.0 or i != 0.0):
            delta = - rescaledLP(x, mp) / dfunc(x, mp)
            plt.plot([x, x + delta], [func(x, mp), 0.], color='black')
            plt.plot([x + delta, x + delta], [0., func(x + delta, mp)], color='black')
            x = x + delta
            if(np.abs(delta) < tol):
                plt.savefig(f"ps-7/plots/minimization{n}")
                return(x)
    plt.savefig(f"ps-7/plots/minimization{n}")

#moon to earth
mp = m_moon/m_earth
n = 1
rp = newton_raphson(mp, n)
r = R_earth2moon * rp
print(f"the lagrange point between the moon and earth  is {r} m")

#sun to earth
mp = m_earth/m_sun
n = 1
rp = newton_raphson(mp, n)
r = R_earth2sun * rp
print(f"the lagrange point between the sun and earth  is {r} m")


#sun to jupiter - earth
mp = m_jupiter/m_sun
n = 1
rp = newton_raphson(mp, n)
r = R_earth2sun * rp
print(f"the lagrange point between the sun and jupiter-earth thing  is {r} m")