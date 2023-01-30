import qutip as qt

# ---> correlation function using QuTiP
corrs_qutip =  qt.correlation_2op_1t(qt.Qobj(H), 
None, times, [qt.Qobj(op) for op in c_ops], qt.Qobj(sp), qt.Qobj(sm))
wlist_qutip, spec_qutip = qt.spectrum_correlation_fft(times, corrs_qutip) # Qutip

# Transition energies
evals,_ = np.linalg.eig(H) # Eigenvalues of the Hamiltonian
bohr_freqs = np.sort(np.array([a-b for a in evals 
                                   for b in evals]))

# ---> Plot
fig, ax = plt.subplots(1,2, figsize = (8,2))
ax[0].plot(times/Omega, np.real(corrs), 'b-', label = 'Real')
ax[0].plot(times/Omega, np.imag(corrs), 'r--', alpha = 0.5, label = 'Imaginary')
ax[0].legend()
ax[0].set_xlabel(r'$t/\Omega$', usetex = True)
ax[0].set_xlim([0,100])
ax[0].set_ylabel(r'$C(t)$  (a.u.)', usetex = True)
ax[0].set_title('Correlation function')
ax[1].plot(wlist, spec, 'k.', markersize = 3, label = 'Semigroup')
ax[1].set_xlabel(r'$\omega/\Omega$',usetex = True)
ax[1].set_ylabel(r'$E(\omega)$  (a.u.)',usetex = True)
ax[1].set_xlim([-2,2])
ax[1].vlines(bohr_freqs,0,18, linewidths = 1, 
             colors = 'black', alpha = 0.2, linestyles = 'dashed')
ax[1].annotate(r'$|g\rangle \to |e\rangle$', (-1.3,6),usetex= True)
ax[1].annotate(r'$|e\rangle \to |g\rangle$', (0.7,6),usetex= True)
ax[1].annotate(r'$|e\rangle \to |e\rangle$', (-0.3,14.8),usetex= True)
ax[1].annotate(r'$|g\rangle \to |g\rangle$', (-0.3,16.6),usetex= True)
ax[1].plot(wlist_qutip,spec_qutip, 
           'k-', label= 'QuTiP', alpha = 0.3)
ax[1].legend()
ax[1].set_title('Emission spectrum');
fig.savefig('figures/correlations.pdf',transparent=True,bbox_inches='tight')
