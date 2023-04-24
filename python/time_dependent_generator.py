import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

omega0,omega,gamma = 2,3,0.3 # system parameters
H0 = omega0*np.array([[1,0],[0,-1]])/2 # Hamiltonian H0
H1 = omega0*np.array([[0,1],[1,0]])/2 # Hamiltonian H1
c_ops = [np.sqrt(gamma)*np.array([[0,0],[0,1]])] # Lindblad operator
L0 = Liouvillian(H0,c_ops) # superoperator L0
L1 = Liouvillian(H1,[]) # superoperator L1 without time dependence
rho0 = np.array([[1,0],[0,0]]) # initial state

# time-dependent coupling
v = lambda t: np.cos(omega*t)

# method for the dynamics of the system
def dynamics(tf,sample):
    # time steps
    times = np.linspace(0,tf,sample)
    # finite difference
    dt = times[1]-times[0]
    # dimension of the system
    d = len(rho0)
    # initialise state in vector form 
    v_rho_t = np.reshape(rho0,(d**2,1))
    # initialise population dynamics
    pops = []
    # propagation
    for t in times:
        # reshape into density operator
        rho_t = np.reshape(v_rho_t,(d,d))
        # append populations
        pops.append([np.real(np.trace(rho_t@rho0))])
        # update superoperator
        superop = L0 + v(t)*L1
        # propagator
        P = expm(superop*dt)
        # propagate state for dt
        v_rho_t = P @ v_rho_t
    # return
    return (times,np.array(pops))

# results
tf = 30
data_hi = dynamics(tf,sample = 1000)
data_mid = dynamics(tf,sample = 50)
data_low = dynamics(tf,sample = 10)
times = np.linspace(0,tf,200)

# plot
fig, (ax,av) = plt.subplots(2,1,figsize = (6,3),gridspec_kw={'height_ratios':[2,1]})
fig.subplots_adjust(hspace=0.1)
ax.plot(*data_hi, 'g-', alpha = 1, label = r'$\delta t$ = '+str(tf/len(data_hi[0])));
ax.plot(*data_mid, 'y.--', alpha = 0.7, label = r'$\delta t$ = '+str(tf/len(data_mid[0])));
ax.plot(*data_low, 'r.--', alpha = 0.5, label = r'$\delta t$ = '+str(tf/len(data_low[0])));
ax.set_ylabel(r'$\mathrm{Tr}[\rho(t)\rho_0]$', usetex=True, fontsize = 10)
ax.set_xticks([])
av.set_xlabel(r'$t$', usetex = True, fontsize = 10)
av.plot(times,v(times),'k-');
av.set_ylabel(r'$v(t)$', usetex = True,fontsize = 10);
ax.legend();
