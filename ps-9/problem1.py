import numpy as np
import scipy

class HarmonicOscillator(object):
    """Class to calculate the Kepler problem
    
    Parameters
    ----------
    
    mass : np.float32
        total mass

    diff : np.float32
        amount for finite difference derivative for potential
    
    Notes
    -----
    
    Calculates derivative from potential instead of force law directly
    (more easily accommodates perturbations to potential)
    
    Set up to use RK4/5 by default
"""
    def __init__(self, w = 1, diff=1.e-7):
        self.w = w
        # self.diff = diff
        self.xdiff = np.array([self.diff, 0., 0.])
        # self.ydiff = np.array([0., self.diff, 0.])
        # self.zdiff = np.array([0., 0., self.diff])
        self.set_ode()
        return 
    def set_ode(self):
        """Setup ODE integrator (RK5)"""
        self.ode = scipy.integrate.ode(self.dwdt)
        self.ode.set_integrator('dopri5') # Runge-Kutta
        return
    # def _diff_central(self, func=None, x=None, dx=None, factor=1.):
    #     """Central difference"""
    #     return((func(x + 0.5 * dx * factor) - func(x - 0.5 * dx * factor)) /
    #            (factor * self.diff))
    # def _diff_correct(self, func=None, x=None, dx=None):
    #     """Higher order difference"""
    #     return((4. * self._diff_central(func=func, x=x, dx=dx, factor=0.5) -
    #             self._diff_central(func=func, x=x, dx=dx)) / 3.)
    def gradient(self, x=None):
        """Returns gradient
        
        Parameters
        ----------
        
        x : ndarray of np.float32
            [3]-d position
        
        Returns
        -------
        
        grad : ndarray of np.float32
            [3]-d gradient of potential
"""
        g = self._diff_correct(func=self.potential, x=x, dx=self.xdiff)
        # g = np.zeros(3)
        # g[0] = self._diff_correct(func=self.potential, x=x, dx=self.xdiff)
        # g[1] = self._diff_correct(func=self.potential, x=x, dx=self.ydiff)
        # g[2] = self._diff_correct(func=self.potential, x=x, dx=self.zdiff)
        return g
    def potential(self, x=None):
        """Returns potential
        
        Parameters
        ----------
        
        x : ndarray of np.float32
            [3]-d position
        
        Returns
        -------
        
        phi : np.float32
            potential
"""
        # r = np.sqrt(x[..., 0]**2 + x[..., 1]**2 + x[..., 2]**2)
        return - self.w ** 2 * x
    def dwdt(self, t, w):
        """Phase space time derivative
        
        Parameters
        ----------
        
        t : np.float32
            current time
            
        w : ndarray of np.float32
            [6] phase space coords (3 position then 3 velocity)
        
        Returns
        -------
        
        dwdt : ndarray of np.float32
            [6] time derivatives to integrate
"""
        x = w[:3]
        v = w[3:]
        dwdt = np.zeros(6)
        dwdt[:3] = v
        dwdt[3:] = - self.gradient(x)
        return(dwdt)
    def energy(self, w=None):
        """Calculate energy of position
        
        Parameters
        ----------
        
        w : ndarray of np.float32
            [..., 3] phase space coords
            
        Returns
        -------
        
        energy : ndarray of np.float32
            energy
"""
        pe = self.potential(w[..., :3])
        ke = 0.5 * (w[..., 3:]**2).sum(axis=-1)
        return(pe + ke)
    def integrate(self, w0=None, t0=0., dt=0.1, nt=100):
        """Integrate the equations
        
        Parameters
        ----------
        
        t0 : np.float32
            initial time
            
        w0 : ndarray of np.float32
            [6] initial phase space coords (3 position then 3 velocity)
            
        dt : np.float32
            time interval to integrate per output
            
        nt : np.int32
            number of intervals
"""
        self.ode.set_initial_value(w0, t0)
        w = np.zeros((nt, 6))
        t = np.zeros(nt)
        w[0, :] = w0
        t[0]= t0
        for indx in np.arange(nt - 1) + 1:
            t[indx] = t[indx - 1] + dt
            self.ode.integrate(self.ode.t + dt)
            w[indx, :] = self.ode.y
        return(t, w)