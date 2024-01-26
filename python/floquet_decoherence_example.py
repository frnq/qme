import numpy as np
import matplotlib.pyplot as plt

# Total Hamiltonian:       H = H0 + Hint*cos(omega*t)
# Bare Hamilotnian:        H0 = 0.5*delta*sigma_z + epsilon*sigma_x 
# Interaction Hamiltonian: Hint = 0.5*Vstr*sigma_z
# Decoherence (dephasing): L = Gammaz * sigma_z

# Define the absorption as the probability of measuring in the state measvec
measvec = np.asarray([1, 0])

# Example values of the Hamiltonian parameters
# energy
epsilon = 0.2
# detuning range
delta_range = np.arange(-6.0, 6.02, 0.02)   
# driving frequency
omega = 1.5;       
# number of photons                        
nphotons = 13                              

# Dephasing operator and rate
Lop = np.asarray([[1,0],[0,-1]])
Gammaz_range = np.asarray([0.005, 0.05, 0.1])

# operator associated with measvec
Aop = np.outer(measvec, measvec)

# Preallocate storage vectors for the absorption spectra and the eigenspectrum
Absorp_av = np.zeros((len(delta_range),len(Gammaz_range)))
spec = np.zeros((2, len(delta_range)))

# loop over dephasing rates to generate spectra
for gc, Gammaz in enumerate(Gammaz_range):  
    # loop over interactions
    for jc, delta in enumerate(delta_range):

        # get the list of Lindblad operators
        Ls = [np.sqrt(Gammaz)*Lop]
        
        # define internal Hamiltonian
        H0 = np.asarray([[delta/2,epsilon],[epsilon,-delta/2]]).astype(complex)
        [evs, evecs] = np.linalg.eigh(np.real(H0)) 
        
        # append spectrum as a function of delta
        spec[:,jc] = evs 
        
        # Use ground state as initial state.
        Rho0 = np.outer(evecs[1], evecs[1])
        # vectorise state
        RhoVec0 = np.reshape(Rho0, (4, 1))
        
        # Interaction part of the Hamiltonian
        Hint = np.asarray([[1,0j],[0j,-1]])/2
        
        # Build up superoperators
        Sys1 = np.eye(len(H0))

        # Bare superoperator (with dephasing operator)
        P0 = Liouvillian(H0, Ls)
        
        # Interaction superoperator (time dependent)
        Pint = Liouvillian(Hint, [])
        
        # timestep
        times = [10/Gammaz]

        # Floquet propagator
        USum = floquet_decoherence(times, omega, nphotons, P0, Pint, average=0)
    
        # Compute propagator in Floquet span and then sum over photon states, 
        # outputting the final summation for density matrix at time times
        Rhot = np.reshape(USum[:,:,0] @ RhoVec0, (2, 2), order='F')
        # Output the probability of measurement results of operator Aop.
        Absorp_av[jc,gc] = np.real(np.trace(Aop @ Rhot))

# plot
fig, ax = plt.subplots(2,1, figsize=(6,4))

# plot spectrum (ground and excited states) as a function of delta
ax[0].plot(delta_range/omega, spec[0,:], 'b', label = r'excited')
ax[0].plot(delta_range/omega, spec[1,:], 'r', label = r'ground')
for mult in np.arange(-3,3.1,1):
    ax[0].axvline(x=mult*omega/omega, ymin=-4, ymax=4, linestyle='dashed',linewidth=1, color = 'black', alpha = 0.5)
ax[0].set_ylim([-4,4])
ax[0].set_ylabel('Energy')
ax[0].legend();
ax[0].set_ylabel(r'Energy ($\lambda$)', fontsize =10);
# plot absorption probability as a function of delta
# over the different dephasing rates
for i in range(3):
    ax[1].plot(delta_range/omega, Absorp_av[:,i], label = r'$\Gamma_z$ = '+ str(Gammaz_range[i]))
ax[1].legend(frameon=False)
ax[1].set_ylim([0,1])
ax[1].set_xlabel(r'detuning  ($\delta/\omega$)', fontsize =10);
ax[1].set_ylabel('Absorption probability', fontsize =10)
for mult in np.arange(-3,3.1,1):
    ax[1].axvline(x=mult*omega/omega, ymin=0, ymax=1, linestyle='dashed',linewidth=1, color = 'black', alpha = 0.5)
fig.savefig('figures/floquet_decoherence.pdf');
