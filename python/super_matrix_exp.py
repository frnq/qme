import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qutip import *

H = Qobj(np.array([[0,1],[1,0.5]])) # Hamiltonian using Qobj class
c_ops = [Qobj(np.array([[0,1],[0,0]]))] # Lindblad operators using Qobj class

superop = liouvillian(H,c_ops) # superoperator using QuTiP
rho0 = ket2dm(basis(2,0)) # initial state

# propgation using exmp from scipy
def Propagate(rho0, superop, t):
    d = len(rho0.full()) # dimension of the system
    propagator = expm(superop.full() * t) # propgator
    vec_rho_t = propagator.dot(np.reshape(rho0.full(),(d**2,1))) # apply to initial state
    return Qobj(np.reshape(vec_rho_t,(d,d))) # return rho(t) using Qobj class

# time steps
times_0, times_1 = np.linspace(0,10,100),  np.linspace(0,10,10)
# Population of the basis state with index i = 0 using:
pops_0 = np.array([ (Propagate(rho0,superop,t)*ket2dm(basis(2,0)) ).tr() for t in times_0]) # expm
pops_1 = mesolve(H,rho0,times_1,c_ops = c_ops, e_ops = [ket2dm(basis(2,0))]).expect[0] # mesolve

# plot
fig, ax = plt.subplots()
ax.plot(times_0, pops_0, 'b-', label = 'w/ expm');
ax.plot(times_1, pops_1, 'ko', label = 'w/ mesolve');
ax.legend();
