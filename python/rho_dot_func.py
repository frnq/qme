import scipy as sp

# function to evaluate the time derivative drho/dt
# that generates the evolution, using sparse arrays
# H, Ls, rho are sparse arrays

def Generator(H,Ls,rho,hbar = 1):
    # initialise the sparse generator
    rho_dot = rho*0
    # add Hamiltonian
    rho_dot += -1j*(H@rho-rho@H)/hbar
    # add Lindblad operators 
    # assuming that the rates are embedded in L
    for L in Ls:
        rho_dot += (L@rho@L.conjugate().T 
                    - (L.conjugate().T @ L @ rho 
                    + rho @ L.conjugate().T @ L)/2)
    # return
    return rho_dot

# Choose dimension
d = 1000

# Define Hamiltonian (1D chain)
row = [k for k in range(d)]+[k for k in range(d)]+[(k+1) % d for k in range(d)]
col = [k for k in range(d)]+[(k+1) % d for k in range(d)]+[k for k in range(d)]
dat = [2 for k in range(d)]+[1 for k in range(d)]+[1 for k in range(d)]
H = sp.sparse.csr_array((dat, (row, col)), shape = (d,d))

# Define Lindblad operators (dephasing)
Ls = []
for k in range(d):
    row = [k]
    col = [k]
    dat = [0.1]
    L = sp.sparse.csr_array((dat, (row, col)), shape = (d,d))
    Ls.append(L)

# Define initial state
row = [0]
col = [0]
dat = [1]
rho = sp.sparse.csr_array((dat, (row,col)), shape = (d,d))

# calculate rho_dot
%time rho_dot = Generator(H,Ls,rho)
