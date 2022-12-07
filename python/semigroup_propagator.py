import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

H = np.array([[0,1],[1,0.5]]) # Hamiltonian
c_ops = [np.array([[0,1],[0,0]])] # Lindblad operators

superop = Liouvillian(H,c_ops) # superoperator
rho0 = np.array([[1,0],[0,0]]) # initial state
    
dt, m = 0.5, 20 # time-step and number of time steps
d = len(rho0) # dimension or the state
P = expm(superop * dt) # propgator
times_0, pops_0 = [], [] # allocate sets
t, rho = 0., rho0 # initialise
# propagate
for k in range(m):
    pops_0.append(np.real(np.trace(rho * rho0))) # append current population
    times_0.append(t) # append current time
    # propagate time and state
    t, rho = t+dt, np.reshape(P.dot(np.reshape(rho,(d**2,1))),(d,d))

# propgation using exmp from scipy
def Propagate(rho0, superop, t):
    d = len(rho0) # dimension of the system
    propagator = expm(superop * t) # propgator
    vec_rho_t = propagator.dot(np.reshape(rho0,(d**2,1))) # apply to initial state
    return np.reshape(vec_rho_t,(d,d)) # return rho(t)

# time steps
times_1 = np.linspace(0,10,100)
# Population dynamics of the initial state
pops_1 = np.array([np.real(np.trace(Propagate(rho0,superop,t)*rho0)) for t in times_1]) # expm

# plot
fig, ax = plt.subplots(figsize=(6,2))
ax.plot(times_0, pops_0, 'ko', label = 'w/ semigroup',fillstyle='none');
ax.plot(times_1, pops_1, 'b-', label = 'w/ expm');
ax.legend();
ax.set_ylabel(r'$\mathrm{Tr}[\rho(t)\rho_0]$', usetex = True, fontsize = 10);
ax.set_xlabel(r't', usetex = True, fontsize = 10);
