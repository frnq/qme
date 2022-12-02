import numpy as np

def is_state(rho):
    evals = np.linalg.eig(rho)[0] # eigenvalues of rho
    non_unit = 1 - np.trace(rho) # deviation from unit trace
    non_herm = np.linalg.norm(np.array([rho[i,j]-
                                        np.conjugate(rho[j,i]) 
                                        for i in range(len(rho)) 
                                        for j in range(i+1,len(rho))])
                             ) # deviation from Hermitianity
    non_pos = np.sum(np.array([(abs(val)-val)/2 
                               for val in evals])
                    ) # deviation from positivity
    # return 1 if rho is a state, less the 1 otherwise  
    return 1-np.linalg.norm(np.array([non_unit,non_herm,non_pos]))
    
# a state    
rho_1 = np.array([[0.2,0,0],[0,0.3,0],[0,0,0.5]])
print(is_state(rho_1))

# a state with some error
rho_2 = rho_1 + np.array([[-1e-4,1e-6,0],[0,0,1e-2j],[1e-3j,0,1e-4]])
print(is_state(rho_2))
