from random import random
import numpy as np
import matplotlib.pyplot as plt

NbigBi = 100000
NTi = 0
NBi = 0
NPb = 0

Pbtau = 3.3 * 60 # half life of lead 209 atom
Titau = 2.2 * 60 # half life of Titanium 209 atom
BigBiTau = 46 * 60

h = 1

pPb2Bi = 1 - 2 ** (-h/Pbtau)# Decay of lead 209 atom
pTi2Pb = 1 - 2 ** (-h/Titau)# Decay of Ti 209 atom
pBigBi2Ti = 1 - 2 ** (-h/BigBiTau)# Decay of Bi 213 atom

tmax = 5000

tpoints = np.arange(0, tmax, h)

BigBipoints = []
Bipoints = []
Pbpoints = []
Tipoints = []


for t in tpoints:
    Pbpoints.append(NPb)
    Bipoints.append(NBi)
    Tipoints.append(NTi)
    BigBipoints.append(NbigBi)

    TiDecay = 0
    Pbdecay = 0
    BigBiDecay = 0
    
    for i in range(NbigBi):
        if random()<pBigBi2Ti:
            BigBiDecay +=1
    
    for i in range(BigBiDecay):
        NbigBi -= 1
        if random() < 2.09 * 10 ** (-2):
            NTi += 1
        else:
            NPb += 1

    for i in range(NTi):
        if random()<pTi2Pb:
            TiDecay +=1
    NTi -= TiDecay
    NPb += TiDecay

    for i in range(NPb):
        if random()<pPb2Bi:
            Pbdecay+=1
    NPb -= Pbdecay
    NBi += Pbdecay



fig, axes = plt.subplots(5, 1, figsize = (15, 30))

axes[0].plot(tpoints, BigBipoints, label = "213 Bi")
axes[0].plot(tpoints, Bipoints, label = "209 Bi")
axes[0].plot(tpoints, Pbpoints, label = "209 Pb")
axes[0].plot(tpoints, Tipoints, label = "209 Ti")
axes[0].set_title("All decays")
axes[0].set_xlabel("Time (s)")
# ax[0]es[1].set_xlabel("Time (s)")

axes[0].set_ylabel("Number of atoms")
# ax[1]es[1].set_ylabel("Number of atoms")
axes[0].legend()

axes[1].plot(tpoints, BigBipoints, label = "213 Bi")
axes[1].set_title("213 Bi count")
axes[1].set_xlabel("Time (s)")
# ax[1]es[1].set_xlabel("Time (s)")2
axes[1].set_ylabel("Number of atoms")
# ax[1]es[1].set_ylabel("Number of atoms")
axes[1].legend()

axes[2].plot(tpoints, Bipoints, label = "209 Bi")
axes[2].set_title("209 Bi count")
axes[2].set_xlabel("Time (s)")
# ax[2]es[1].set_xlabel("Time (s)")2
axes[2].set_ylabel("Number of atoms")
# ax[2]es[1].set_ylabel("Number of atoms")
axes[2].legend()

axes[3].plot(tpoints, Pbpoints, label = "209 Pb")
axes[3].set_title("209 Pb count")
axes[3].set_xlabel("Time (s)")
# ax[3]es[1].set_xlabel("Time (s)")2
axes[3].set_ylabel("Number of atoms")
# ax[3]es[1].set_ylabel("Number of atoms")
axes[3].legend()

axes[4].plot(tpoints, Tipoints, label = "209 Ti")
axes[4].set_title("209 Ti count")
axes[4].set_xlabel("Time (s)")
# ax[4]es[1].set_xlabel("Time (s)")2
axes[4].set_ylabel("Number of atoms")
# ax[4]es[1].set_ylabel("Number of atoms")
axes[4].legend()

plt.savefig("ps-3/plots/decay")

