from scipy.linalg import null_space

# two-level Hamiltonian from delta and omega parameters
def H_tls(omega, delta):
    return np.array([[0,omega],[omega,delta]])

# Lindblad operators for spontaneous relaxation and dephasing
def Ls_tls(g_relax, g_deph):
    return [np.sqrt(g_relax) * np.array([[0,1],[0,0]]), np.sqrt(g_deph) * np.array([[0,0],[0,1]])]

# ------- (i) relaxation with driving and no dephasing --------
omega, delta, g_relax, g_deph = 1, 1, 1, 0
superop = Liouvillian(H_tls(omega,delta),Ls_tls(g_relax,g_deph)) # Liouville superoperator
null = null_space(superop)
print('(i) The steady-state space is a linear subpace of dimension = '+str(len(null.T)-1))
rho_ss = np.reshape(null, (2,2) )
rho_ss /= np.trace(rho_ss)

# ------- (ii) dephasing with no driving --------
omega, delta, g_relax, g_deph = 0, 1, 0, 1
superop = Liouvillian(H_tls(omega,delta),Ls_tls(g_relax,g_deph)) # Liouville superoperator
null = null_space(superop)
print('(ii) The steady-state space is a linear subpace of dimension = '+str(len(null.T)-1))
