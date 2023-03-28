from qutip import *

psi = (tensor(basis(2,0),basis(2,0)) + tensor(basis(2,1),basis(2,1))).unit() # normalised Bell state
rho = psi * psi.dag() # or rho = ket2dm(psi)
rho_1, rho_2 = rho.ptrace(0), rho.ptrace(1) # marginal states using partial trace
