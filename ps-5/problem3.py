import numpy as np
import matplotlib.pyplot as plt

fName = "ps-5/signal.dat"

data = np.loadtxt(fName, delimiter='|', skiprows=1, usecols=(1, 2))

time = data[:, 0]
signal = data[:, 1]

print("Time array:", time)
print("Signal array:", signal)



plt.scatter(time, signal)
plt.savefig("ps-5/plots/signal")

A = np.zeros((len(time), 4))
A[:, 0] = 1.
A[:, 1] = time 
A[:, 2] = time**2
A[:, 3] = time**3

(u, w, vt) = np.linalg.svd(A, full_matrices=False)
weff = np.where(w != 0, w, np.inf)

ainv = vt.transpose().dot(np.diag(1. / weff)).dot(u.transpose())
# ainv = np.linalg.pinv(A)
teff = ainv.dot(signal)

signaleff = A.dot(teff)

plt.scatter(time, signal, label = "raw data")
plt.scatter(time, signaleff, label = "3rd order fit")
plt.xlabel("time")
plt.ylabel("signal")
plt.legend()

plt.savefig("ps-5/plots/signalfit")

residuals = signaleff - signal
fig, ax = plt.subplots()

ax.scatter(time, residuals)
ax.set_xlabel("time")
ax.set_ylabel("residuals")
# ax.legend()

plt.savefig("ps-5/plots/residuals")
print(f"std of residuals: {np.std(residuals)}")


poly = 8
A = np.zeros((len(time), poly))
for p in range(poly):
    A[:, p] = time ** p


(u, w, vt) = np.linalg.svd(A, full_matrices=False)
weff = np.where(w != 0, w, np.inf)
conditionNumber = weff.max()/w.min()
print(f"condition number of {poly}th order polynomial: {conditionNumber}")
ainv = vt.transpose().dot(np.diag(1. / weff)).dot(u.transpose())
# ainv = np.linalg.pinv(A)
teff = ainv.dot(signal)

signaleff = A.dot(teff)
fig, ax = plt.subplots()
ax.scatter(time, signal, label = "raw data")
ax.scatter(time, signaleff, label = "8th order fit")
ax.set_xlabel("time")
ax.set_ylabel("signal")
ax.legend()

plt.savefig("ps-5/plots/signalfithighpoly")

residuals = signaleff - signal
fig, ax = plt.subplots()

ax.scatter(time, residuals)
ax.set_xlabel("time")
ax.set_ylabel("residuals")
# ax.legend()

plt.savefig("ps-5/plots/residualshighPoly")
print(f"std of residuals: {np.std(residuals)}")


# poly = 5
harmonicNumber = 500
A = np.zeros((len(time), 2*harmonicNumber+1))
period = time.max()/3
omega = 2*np.pi/period

for i in range(harmonicNumber):
    A[:, i] =  np.sin(omega*(i+1) * time)
    A[:, harmonicNumber+i] =  np.cos(omega * (i+1) * time)
A[:, harmonicNumber*2] = 1


(u, w, vt) = np.linalg.svd(A, full_matrices=False)
weff = np.where(w != 0, w, np.inf)
conditionNumber = weff.max()/w.min()
print(f"condition number of sinuosoidal series: {conditionNumber}")
ainv = vt.transpose().dot(np.diag(1. / weff)).dot(u.transpose())
# ainv = np.linalg.pinv(A)
teff = ainv.dot(signal)

signaleff = A.dot(teff)
fig, ax = plt.subplots()
ax.scatter(time, signal, label = "raw data")
ax.scatter(time, signaleff, label = "sine and cosine fit")
ax.set_xlabel("time")
ax.set_ylabel("signal")
ax.legend()

plt.savefig("ps-5/plots/signalfitsin")

residuals = signaleff - signal
fig, ax = plt.subplots()

ax.plot(time, residuals)
ax.set_xlabel("time")
ax.set_ylabel("residuals")
# ax.legend()

plt.savefig("ps-5/plots/residualshighPoly")
print(f"std of residuals: {np.std(residuals)}")