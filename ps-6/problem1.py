import astropy.io.fits
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
hdu_list = astropy.io.fits.open('ps-6/specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

fig, ax = plt.subplots()

n = 4
ax.set_xlabel("wavelength ($log_{10}\lambda$)")
ax.set_ylabel("Flux ($10^{-17} erg s^{-1} cm^{-2} A^{-1}$)")
ax.set_title(f"galaxy {n} flux vs wavelength")
ax.plot(logwave, flux[n])

plt.savefig("ps-6/plots/galaxy")

wave = 10 ** logwave 

flux_integral = trapz(flux, wave)

flux_norm = []
for i in range(len(flux_integral)):
    flux_norm.append(np.array(flux[i]/flux_integral[i]))

flux_norm = np.array(flux_norm)
# means = []
# for i in range(len(flux_norm)):
#     mean = np.mean(flux_norm[[i]])
#     flux_norm[i] = flux_norm[i] - mean
#     means.append(mean)

means = np.mean(flux_norm, axis = 0)
flux_means = (means[:, np.newaxis] * np.ones((1, 9713))).T

Ngal = len(flux)

R = ((1/Ngal) * (flux_norm - flux_means)).T





C =np.dot(R, R.T)

eigenvalues, eigenvectors = np.linalg.eig(C)

fig, ax = plt.subplots()

for i in range(5):
    ax.plot(logwave, eigenvectors[i], label = f"eigenvector with eval {eigenvalues[i]}")
ax.set_xlabel("wavelength ($log_{10}\lambda$)")
ax.set_ylabel("Flux ($10^{-17} erg s^{-1} cm^{-2} A^{-1}$)")
ax.set_title("First 5 eigenvectors")
ax.legend()

plt.savefig("ps-6/plots/eigenvectors")

(u, w, vt) = np.linalg.svd(R, full_matrices=False)
weff = np.where(w != 0, w, np.inf)
conditionNumber = weff.max()/w.min()
print(f"condition number of sinuosoidal series: {conditionNumber}")
# Rinv = vt.transpose().dot(np.diag(1. / weff)).dot(u.transpose())
# # ainv = np.linalg.pinv(A)
# teff = ainv.dot(signal)

# signaleff = A.dot(teff)

print(vt- eigenvectors)
# # fig, ax = plt.subplots()

# ax.plot()