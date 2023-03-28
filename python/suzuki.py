import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

sx = np.array([[0,1],[1,0]]) # sigma_x
sy = np.array([[0,-1j],[1j,0]]) # sigma_y
sp = (sx+1j*sy)/2 # sigma_plus
sm = (sx-1j*sy)/2 # sigma_minus

superop_1 = Liouvillian(sx,[sp]) # superoperator 1 
superop_2 = Liouvillian(sy,[sm]) # superoperator 2 
rho0 = np.array([[1,0],[0,0]]) # initial state

# plot
fig, ax = plt.subplots(figsize = (6,2))
alpha = [0.2,0.5,1]
    
# number of time steps m for three possible values of m = 50,100 and 1000
for km,m in enumerate([50,100,1000]):
    dt = 10/m # time-step
    d = len(rho0) # dimension of the state
    P_1, P_2 = expm(superop_1 * dt), expm(superop_2 * dt) # propgators
    P = P_1.dot(P_2) # suzuki-trotter expansion for small times dt
    times_0, pops_0 = [], [] # allocate sets
    t, rho = 0., rho0 # initialise
    # propagate
    for k in range(m):
        pops_0.append(np.trace(rho * rho0)) # append current population
        times_0.append(t) # append current time
        # propagate time and state
        t, rho = t+dt, np.reshape(P.dot(np.reshape(rho,(d**2,1))),(d,d))
    # plot    
    ax.plot(times_0, np.real(pops_0), 'b-', label = r'$m$ = '+str(m), alpha = alpha[km]);

# time steps
times_1 = np.linspace(times_0[0],times_0[-1],20)
# plot
ax.legend();
ax.set_ylabel(r'$\mathrm{Tr}[\rho(t)\rho_0]$', usetex = True, fontsize = 10)
ax.set_xlabel(r'$t$', usetex = True, fontsize = 10);
