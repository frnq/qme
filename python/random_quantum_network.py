import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from qutip import *

# constants and units
fs = (2.4189e-2)**-1  # femtosecond in Hartree AU
eV = ( fs * 0.6582 )**-1  #electronvolt in Hartree AU

np.random.seed(1) # fix random seed
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
H = (Qobj(np.diag(Es)) 
     + sum([V*basis(N,i)*basis(N,j).dag() 
            + np.conjugate(V)*basis(N,j)*basis(N,i).dag() 
            for i,j,V in Vs]))

def S(w,wc,eta,beta,thresh = 1e-10): # Noise Power Spectum
    return (2*np.pi*eta*w*np.exp(-abs(w)/wc) /
            (1-np.exp(-w*beta)+thresh)*(w>thresh or w<=-thresh) +
            2*np.pi*eta*beta**-1*(-thresh<w<thresh)) 

# function for NPS(w)
NPS = lambda w: S(w,wc=150e-3*eV,eta=1e-1,beta=(25e-3*eV)**-1)

a_ops = [[ket2dm(basis(N,k)), NPS]  for k in range(N)] # coupling operators
R, ekets = bloch_redfield_tensor(H, a_ops) # Bloch-Redfield tensor
X = Qobj(np.diag([k for k in range(N)])) # "position" operator
X_vec = operator_to_vector(X.transform(ekets))
rho0 = ket2dm(basis(N,0)).unit() # initial state
rho_vec = operator_to_vector(rho0.transform(ekets))
dts = [1e-1*fs,1e1*fs,1e3*fs] # adaptive time-scales
t,tf = 0,1e6*fs
times, pos = [], []
for dt in dts:
    P = Qobj(expm(R.full()*dt), dims = R.dims)
    for m in range(1000):
        times.append(t) # append time 
        pos.append( (rho_vec.dag()*X_vec).tr() ) # append position
        rho_vec = P*rho_vec # propagate state
        t += dt # propagate time
# plot
plt.plot(np.array(times)/fs, np.array(pos), 'k-')
plt.xscale('log')
