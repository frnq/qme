import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# constants and units
fs = (2.4189e-2)**-1  # femtosecond in Hartree AU
eV = ( fs * 0.6582 )**-1  #electronvolt in Hartree AU

np.random.seed(0) # fix random seed
N = 10 # number sites
sigma_E, sigma_V = 100e-3 * eV, 50e-3 * eV # disorder parameters

# random energies
Es = np.random.normal(0,sigma_E,N) 
# random couplings
all_pairs = [(x,y) for x in range(N) for y in range(x+1,N)]
indices = np.random.choice([k for k in range(len(all_pairs))], N, replace = False)
pairs = [all_pairs[k] for k in indices]
Vs = [(i,j,np.random.normal(0,sigma_V)) for i,j in pairs]

# Hamiltonian
H = (np.diag(Es)
     + sum([V*np.outer(np.eye(N)[i],np.eye(N)[j])
            + np.conjugate(V)*np.outer(np.eye(N)[j],np.eye(N)[i]) 
            for i,j,V in Vs]))

# Hamiltonian eigendecomposition
evals,ekets = np.linalg.eig(H) # HS's basis
# sort basis
_zipped = list(zip(evals, range(len(evals))))
_zipped.sort()
evals, perm = list(zip(*_zipped))
ekets = np.array([ekets[:, k] for k in perm])
evals = np.array(evals)

def S(w,wc,eta,beta,thresh = 1e-10): # Noise Power Spectum
    return (2*np.pi*eta*w*np.exp(-abs(w)/wc) /
            (1-np.exp(-w*beta)+thresh)*(w>thresh or w<=-thresh) +
            2*np.pi*eta*beta**-1*(-thresh<w<thresh)) 

# function for NPS(w)
NPS = lambda w: S(w,wc=150e-3*eV,eta=1e-1,beta=(25e-3*eV)**-1)

# coupling operators and associated noise power spectra
a_ops = [[np.diag([float(i == k) for i in range(N)]), NPS] for k in range(N)]

# Bloch-Redfield tensor in H basis
R = BR_tensor(H, a_ops)

# position operator
X = np.diag([k for k in range(N)])
X_vec = np.reshape(ekets.conjugate()@X@ekets.T,(1,N**2))
rho0 = np.array([[float(i==0 and j==i) for i in range(N)] for j in range(N)])# initial state
rho_vec = (np.reshape(ekets.conjugate()@rho0@ekets.T,(1,N**2))).T

dts = [1e-1*fs,1e1*fs,1e3*fs] # adaptive time-scales
t,tf = 0,1e6*fs # initialised time, final time
times, pos = [], [] # time, position sets
for dt in dts: # loop over time scales
    P = expm(R*dt) # calculate propagator
    for m in range(1000):
        times.append(t) # append time 
        pos.append( np.real(np.trace((rho_vec.conjugate()@X_vec))) ) # append position
        rho_vec = P@rho_vec # propagate state
        t += dt # propagate time
