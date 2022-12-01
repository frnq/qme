import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qutip import *

H = Qobj(np.array([[0,1],[1,0.5]])) # Hamiltonian using Qobj class
c_ops = [Qobj(np.array([[0,1],[0,0]]))] # Lindblad operators using Qobj class

superop = liouvillian(H,c_ops) # superoperator using QuTiP
rho0 = ket2dm(basis(2,0)) # initial state
    
dt, m = 0.5, 20 # time-step and number of time steps
d = len(rho0.full()) # dimension or the state
P = expm(superop.full() * dt) # propgator
times_0, pops_0 = [], [] # allocate sets
t, rho = 0., rho0 # initialise
# propagate
for k in range(m):
    pops_0.append( (rho * ket2dm(basis(2,0)) ).tr() ) # append current population
    times_0.append(t) # append current time
    # propagate time and state
    t, rho = t+dt, Qobj(np.reshape(P.dot(np.reshape(rho.full(),(d**2,1))),(d,d)))

# propgation using exmp from scipy
def Propagate(rho0, superop, t):
    d = len(rho0.full()) # dimension of the system
    propagator = expm(superop.full() * t) # propgator
    vec_rho_t = propagator.dot(np.reshape(rho0.full(),(d**2,1))) # apply to initial state
    return Qobj(np.reshape(vec_rho_t,(d,d))) # return rho(t) using Qobj class

# time steps
times_1 = np.linspace(0,10,100)
# Population of the basis state with index i = 0 using:
pops_1 = np.array([ (Propagate(rho0,superop,t)*ket2dm(basis(2,0)) ).tr() for t in times_1]) # expm

# plot
fig, ax = plt.subplots()
ax.plot(times_0, pops_0, 'ko', label = 'w/ semigroup',fillstyle='none');
ax.plot(times_1, pops_1, 'b-', label = 'w/ expm');
ax.legend();
