from qutip import *

H = Qobj(H) # Hamiltonian using Qobj class
c_ops = [Qobj(c_op) for c_op in c_ops] # Lindblad operators using Qobj class

superop = liouvillian(H,c_ops) # superoperator
rho0 = Qobj(rho0) # initial state

# time steps
times_1 = np.linspace(0,10,10)
# Population of the basis state with index i = 0 using:
pops_1 = mesolve(H,rho0,times_1,c_ops = c_ops, e_ops = [rho0]).expect[0] # mesolve

# plot
fig, ax = plt.subplots()
ax.plot(times, pops, 'b-', label = 'w/ expm');
ax.plot(times_1, pops_1, 'ko', label = 'w/ mesolve');
ax.set_ylabel(r'Population $\rho_{00}(t)$')
ax.set_xlabel(r't')
ax.legend();
ax.set_aspect(7);
fig.savefig('figures/propagation.png',transparent=True, dpi = 600,bbox_inches='tight')
