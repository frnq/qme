import numpy as np
from scipy.linalg import expm

H = np.array([[0,1],[1,0.5]]) # Hamiltonian
c_ops = [np.array([[0,1],[0,0]])] # Lindblad operators

superop = Liouvillian(H,c_ops) # superoperator
rho0 = np.array([[1,0],[0,0]]) # initial state

# propgation using exmp from scipy
def Propagate(rho0, superop, t):
    d = len(rho0) # dimension of the system
    propagator = expm(superop * t) # propgator
    vec_rho_t = propagator @ np.reshape(rho0,(d**2,1)) # apply to initial state
    return np.reshape(vec_rho_t,(d,d)) # return rho(t)

# time steps
times = np.linspace(0,10,100)
# Population of rho0 in time with expm
pops = np.array([ np.real(np.trace(Propagate(rho0,superop,t)@rho0)) for t in times])
