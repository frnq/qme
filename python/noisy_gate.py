import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.linalg import fractional_matrix_power as matrix_power

# Fidelity between two operators
def Fidelity(a,b):
    ra = matrix_power(a,0.5)
    return np.trace(matrix_power(ra@b@ra,0.5))**2

# Pauli operators
sx,sy,sz = np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.array([[1,0],[0,-1]])
# internal Hamiltonian
w0 = 1
H0 = w0*sz/2
# decomposition of the internal Hamiltonian
vals,kets = np.linalg.eig(H0)
# Hamiltonian to implement the Hadamard gate
H1 = np.eye(2)/2-(sx+sz)/(2*np.sqrt(2))
# Hamiltonian to implement the CNOT gate
H2 = np.kron((np.eye(2)-sz)/2,np.eye(2)/2-sx/2)

# Noise Power Spectum
def S(w,wc,eta,beta,thresh = 1e-10):
    return (2*np.pi*eta*w*np.exp(-abs(w)/wc) /
            (1-np.exp(-w*beta)+thresh)*(w>thresh or w<=-thresh) +
            2*np.pi*eta*beta**-1*(-thresh<w<thresh)) 

# Output state from input state as a function of the thermal energy
def output(rho_in,th_en):
    # dimension
    dim = len(rho_in)
    # function for the noise power spectrum
    gamma = lambda w: S(w,wc=w0/2,eta=1e-3,beta=1/th_en)
    # jump operators
    c_ops1 = [np.sqrt(gamma(val2-val1))*np.kron(np.outer(kets[k1],kets[k2]),np.eye(2)) 
              for k1,val1 in enumerate(vals) for k2,val2 in enumerate(vals)]
    c_ops2 = [np.sqrt(gamma(val2-val1))*np.kron(np.eye(2),np.outer(kets[k1],kets[k2])) 
              for k1,val1 in enumerate(vals) for k2,val2 in enumerate(vals)]
    c_ops = c_ops1+c_ops2
    # evolution time
    tau = np.pi
    # evolution through frist gate
    L1 = Liouvillian(np.kron(H1,np.eye(2)),c_ops)
    rho1 = np.reshape(expm(L1*tau)@np.reshape(rho_in,(dim**2,1)),(dim,dim))
    # evolution through second gate
    L2 = Liouvillian(H2,c_ops)
    rho2 = np.reshape(expm(L2*tau)@np.reshape(rho1,(dim**2,1)),(dim,dim))
    # output state
    return rho2

# initial state
g,e = np.array([1,0]),np.array([0,1])
psi0 = np.kron(g,g)
rho0 = np.outer(psi0,psi0)
# target state
psi_target = (np.kron(g,g)+np.kron(e,e))/np.sqrt(2)
rho_target = np.outer(psi_target,psi_target)

# plot
th_ens = np.logspace(-2,4,100)
fids = np.array([Fidelity(rho_target,output(rho0,th_en)) for th_en in th_ens])
purs = np.array([np.trace(matrix_power(output(rho0,th_en),2)) for th_en in th_ens])
fig, ax = plt.subplots( figsize = (4,2))
ax.plot(th_ens,fids,'k-', label = 'Fidelity');
ax.plot(th_ens,purs,'b--', label = 'Purity', alpha = 0.5);
ax.set_ylim([0.15,1.1])
ax.set_xscale('log')
ax.set_xlabel(r'Thermal energy $k_B T/\omega_0$');
ax.legend();
