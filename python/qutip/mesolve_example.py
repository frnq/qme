import matplotlib.pyplot as plt
from qutip import *

H = Qobj(H) # Hamiltonian using Qobj class
c_ops = [Qobj(c_op) for c_op in c_ops] # Lindblad operators using Qobj class

superop = liouvillian(H,c_ops) # superoperator
rho0 = Qobj(rho0) # initial state

# time steps
times_1 = np.linspace(0,10,10)
# Population of rho0 in time using mesolve
pops_1 = mesolve(H,rho0,times_1,c_ops = c_ops, e_ops = [rho0]).expect[0]

# plot
fig, ax = plt.subplots(figsize=(6,2))
ax.plot(times, pops, 'b-', label = 'w/ expm');
ax.plot(times_1, pops_1, 'k.', label = 'w/ mesolve');
ax.set_ylabel(r'$\mathrm{Tr}[\rho(t)\rho_0]$', usetex=True, fontsize = 10);
ax.set_xlabel(r'$t$', usetex=True, fontsize = 10);
ax.legend();
fig.savefig('figures/propagation.pdf',transparent=True,bbox_inches='tight')
