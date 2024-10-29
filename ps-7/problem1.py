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
    fig, ax = plt.subplots()
    ax.set_title(f"{n}")
    ax.set_xlabel("rescaled r coordinates (a.u)")
    ax.set_ylabel("magnitude of rescaled lagrangian function (a.u)")
    ax.plot(rgrid, rescaledLP(rgrid, mp), label = "function")
    ax.plot(rgrid, 0. * rescaledLP(rgrid, mp), label = "zero value")
    ax.set_ylim((-3, 3))
    for i in np.float32(np.arange(begiter, maxiter))/(maxiter-begiter):
        if(i != 1.0 or i != 0.0):
            delta = - rescaledLP(x, mp) / dfunc(x, mp)
            ax.plot([x, x + delta], [func(x, mp), 0.], color='black', label = "minimization routine")
            ax.plot([x + delta, x + delta], [0., func(x + delta, mp)], color='black', label = "minimization routine")
            x = x + delta
            if(np.abs(delta) < tol):
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[:3], labels[:3])  # Only the first 3 labels
                plt.savefig(f"ps-7/plots/minimization{n}")
                return(x)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3])  # Only the first 3 labels
    plt.savefig(f"ps-7/plots/minimization{n}")

#moon to earth
mp = m_moon/m_earth
rp = newton_raphson(mp, "moon earth")
r = R_earth2moon * rp
print(f"the lagrange point between the moon and earth  is {r} m")

#sun to earth
mp = m_earth/m_sun
rp = newton_raphson(mp,"earth sun")
r = R_earth2sun * rp
print(f"the lagrange point between the sun and earth  is {r} m")


#sun to jupiter - earth
mp = m_jupiter/m_sun
rp = newton_raphson(mp, "jupiter-earth sun")
r = R_earth2sun * rp
print(f"the lagrange point between the sun and jupiter-earth thing  is {r} m")