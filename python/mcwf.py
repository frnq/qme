import numpy as np
import matplotlib.pyplot as plt

sx = np.array([[0,1],[1,0]])
sz = np.array([[1,0],[0,-1]])
psi0 = np.array([1,1])/np.sqrt(2)
H = sz # hamiltonian
Ls = [0.5*sz,0.2*sx] # lindblad operators
Heff = H - 1j/2 * sum([L.T.conjugate()@L for L in Ls]) # effective Hamiltonian
dt = dt = 0.1/np.linalg.norm(H, ord ='fro') # timestep
m = 200 # number of steps
tf = dt*m # final time
times = np.linspace(0,tf,m)

sample = 100 # number of samples
mean = np.zeros(m, dtype = complex) #array for the results
for count in range(sample):
    t = 0
    waves = [psi0]
    for t in times[1:]:
        # generate a random number in (0,1]
        u = np.random.random()
        # array of jump probabilities
        dps = [np.real(dt * (waves[-1].T.conjugate()@(L.T.conjugate()@L)@waves[-1])) for L in Ls]
        # renormalisation factor 1-dP
        dP = np.sum(dps)
        # test
        if dP < u:
            temp = (np.eye(len(psi0))-1j*Heff.T.conjugate()*dt)@(waves[-1])
        else:
            # new random number
            u = np.random.random()
            Q = np.cumsum(dps)/dP
            # pick the jump that has occurred
            k = np.searchsorted(Q, u, side = 'left')
            temp = Ls[k]@waves[-1]
        waves.append(temp/np.linalg.norm(temp))
    mean += np.array([wave.T.conjugate()@sx@wave for wave in waves])
    
mean = np.array(mean)/sample

fig, ax = plt.subplots()
ax.plot(times, np.real(mean), 'b-', alpha = 0.5, label = 'w/ mcwf');
ax.legend();
