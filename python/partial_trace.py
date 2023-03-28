import numpy as np
import matplotlib.pyplot as plt

# Purity of state rho
def Purity(rho):
    return np.trace(rho.dot(rho))

# Partial trace of bipartite systems
def PartialTrace(rho,d1,d2,system=1):
    axis1,axis2 = 0,2
    if system == 2:
        axis1 += 1
        axis2 += 1
    return np.trace(rho.reshape(d1,d2,d1,d2), axis1=axis1, axis2=axis2)

d1,d2 = 2,2 # dimension of each subsystem
B1,B2 = np.eye(d1),np.eye(d2) # basis for each subssystem
thetas = np.linspace(0,np.pi/2,100) # angle for superposition coefficient
purity = [] # purity set
for theta in thetas: # iterate over theta
    psi = (np.cos(theta)*np.kron(B1[0],B2[0])+np.sin(theta)*np.kron(B1[1],B2[1])) # state vector
    rho = np.outer(psi,psi.conjugate()) # density operator associated to psi
    rho1 = PartialTrace(rho,d1,d2,system=1) # marginal state of system 1
    purity.append(Purity(rho1)) # calculate and append purity
fig,ax = plt.subplots(figsize = (6,2))
ax.plot(thetas/np.pi,purity, color = 'blue');
ax.set_xlabel(r'$\theta/\pi$', usetex = True, fontsize = 10);
ax.set_ylabel(r'Purity $\mathcal{P}[\rho_1(\theta)]$', usetex = True, fontsize = 10);
