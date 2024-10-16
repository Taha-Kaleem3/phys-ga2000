import astropy.io.fits
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid as trapz
import timeit

# galaxy_num = 30

hdu_list = astropy.io.fits.open('ps-6/specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data
wave = 10 ** logwave

fig, ax = plt.subplots()

n = 4
ax.set_xlabel("wavelength (Angstroms)")
ax.set_ylabel("Flux ($10^{-17} erg s^{-1} cm^{-2} A^{-1}$)")
ax.set_title(f"galaxy {n} flux vs wavelength")
ax.plot(wave, flux[n])

plt.savefig("ps-6/plots/galaxy")



flux_integral = trapz(flux, wave)

flux_norm = []
for i in range(len(flux_integral)):
    flux_norm.append(np.array(flux[i]/flux_integral[i]))

flux_norm = np.array(flux_norm)
means = []
for i in range(len(flux_norm)):
    mean = np.mean(flux_norm[[i]])
    flux_norm[i] = flux_norm[i] - mean
    means.append(mean)

means = np.mean(flux_norm, axis = 0)
flux_means = (means[:, np.newaxis] * np.ones((1, 9713))).T

Ngal = len(flux)

t1 = timeit.default_timer()

R = ((flux_norm - flux_means))
C =np.dot(R.T, R)
eigenvalues, eigenvectors = np.linalg.eig(C)

t2 = timeit.default_timer()
print(f"Eig took {t2-t1} seconds")
print(f"Eig has a condition number: {np.abs(eigenvalues.max()/eigenvalues.min())} ")
# inx = np.argsort(eigenvalues)[::-1]
# eigenvalues = eigenvalues[inx]
# eigenvectors = eigenvectors[inx]

fig, ax = plt.subplots()

for i in range(5):
    ax.plot(wave, eigenvectors[i], label = f"eigenvector with eval {eigenvalues[i]}")
ax.set_xlabel("wavelength (Angstroms)")
ax.set_ylabel("normalized flux")
ax.set_title("First 5 eigenvectors")
ax.legend()

plt.savefig("ps-6/plots/eigenvectors")


t1 = timeit.default_timer()
(u, w, vt) = np.linalg.svd(R, full_matrices = False)
weff = np.where(w != 0, w, np.inf)
conditionNumber = weff.max()/w.min()
t2 = timeit.default_timer()

print(f"SVD took {t2-t1} seconds")

print(f"condition number of  svd: {conditionNumber}")

# Rinv = vt.transpose().dot(np.diag(1. / weff)).dot(u.transpose())
# # # ainv = np.linalg.pinv(A)
# waveEffSVD = Rinv.dot(flux)
# fluxEffSVD = R.dot(waveEffSVD)

# waveEffEIG = R.T.dot(flux)
# fluxEffEig = R.dot(waveEffEIG)


# fig, ax = plt.subplots()

for n in range(5):
    fig, ax = plt.subplots(2, 1, figsize = (15, 10))

    ax[0].set_xlabel("wavelength (Angstroms)")
    ax[0].set_ylabel("normalized flux")
    ax[0].set_title(f"galaxy {n} flux vs wavelength")
    ax[0].plot(wave, eigenvectors.T[n], label = "SVD", color = "b")
    ax[0].legend()

    ax[1].set_xlabel("wavelength (Angstroms)")
    ax[1].set_ylabel("normalized flux")
    ax[1].set_title(f"galaxy {n} flux vs wavelength")
    ax[1].plot(wave, vt[n], label = "Eig", color = "c")
    ax[1].legend()
    plt.savefig(f"ps-6/plots/galaxy_{n}_SVD_Eig")

inx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[inx]
eigenvectors = eigenvectors[:, inx]

EigV5 = eigenvectors[:, 0:5]
c5 = np.dot(R, EigV5)
fluxEff = np.dot(c5, EigV5.T) * flux_norm + flux_means
for i in range(5):
    fig, ax = plt.subplots(2, 1, figsize = (15, 10))

    ax[0].set_xlabel("wavelength (Angstroms)")
    ax[0].set_ylabel("normalized flux")
    ax[0].set_title(f"galaxy {i} flux vs wavelength")
    ax[0].plot(wave, fluxEff[i], label = "Effective", color = "b")
    ax[0].legend()

    ax[1].set_xlabel("wavelength (Angstroms)")
    ax[1].set_ylabel("flux (not normalized)")
    ax[1].set_title(f"galaxy {i} flux vs wavelength")
    ax[1].plot(wave, flux[i], label = "raw", color = "c")
    ax[1].legend()
    plt.savefig(f"ps-6/plots/Effective_Flux_{i}")


fig, ax = plt.subplots(2, 1, figsize = (15, 10))

ax[0].set_xlabel("c1")
ax[0].set_ylabel("c0")
ax[0].set_title(f"c0 vs c1")
ax[0].scatter(c5.T[1], c5.T[0], color = "b")
ax[0].legend()

ax[1].set_xlabel("c2")
ax[1].set_ylabel("c0")
ax[1].set_title(f"c0 vs c2")
ax[1].scatter(c5.T[2], c5.T[0], color = "b")
ax[1].legend()
plt.savefig(f"ps-6/plots/cvsc")
# print(f"vt")
# print(vt)
# print("evec")
# print(eigenvectors)
# print("vt - evecs")
# print(u- eigenvalues)
# # fig, ax = plt.subplots()

# ax.plot()
residualsSquared = []
Nc = np.arange(1, 21)
for i in range(20):
    EigV5 = eigenvectors[:, 0:i]
    c5 = np.dot(R, EigV5)
    fluxEff = np.dot(c5, EigV5.T)

    residual = (fluxEff - flux) ** 2
    print(np.mean(residual))
    residualsSquared.append(np.mean(residual))
    n = 20

residualsSquared = np.array(residualsSquared)
rms = np.sqrt(1/n * (residualsSquared))

fig, ax = plt.subplots()

ax.set_xlabel("Nc")
ax.set_ylabel("rms residuals")
ax.set_title(f"galaxy residuals")
ax.plot(Nc, rms)

plt.savefig("ps-6/plots/residual")
