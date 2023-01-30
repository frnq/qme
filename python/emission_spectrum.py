import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.linalg import expm, null_space

Omega, Gamma = 1,0.1 # Rabi frequency and decay rate                       
sm = np.array([[0., 1.], [0., 0.]]) # emission operator           
sp = sm.conj().T # absorption operator
H = np.array([[0, Omega], [Omega, 0]])/2 # Hamiltonian
c_ops = [np.sqrt(Gamma)*sm] # collapse operators

L = Liouvillian(H,c_ops) # Liouvillian
rho_ss = np.reshape(null_space(L), (2,2) ) # steady state
rho_ss /= np.trace(rho_ss) # normalised steady state
dim = len(rho_ss) # system dimension

# ---> correlation function 
N = 2000 # samples
times = np.linspace(0.0, 500., N) # time interval              
corrs = np.zeros(N, dtype = complex) # correlation array
dt = times[1]-times[0] # time-step finite difference
P = expm(L * dt) # propagator
B_ss = sm@rho_ss # emission operator applied to steady state
vec_B_ss = np.reshape(B_ss,(dim**2,1)) # vector form of B_ss 

# calculate correlation function over the time interval
for kt,t in enumerate(times):
    corrs[kt] = np.trace(sp@np.reshape(vec_B_ss,(dim,dim))) # collect correlation
    vec_B_ss = P.dot(vec_B_ss) # propagate operator using semigroup composition rule
    
# ---> Obtain correlation spectrum using discrete fourier transform
spec = 2 * np.real(fft(corrs)) * dt # spectrum
wlist = 2 * np.pi*fftfreq(N, dt) # angular frequencies
