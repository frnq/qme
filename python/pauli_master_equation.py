# construct W tensor
W = np.zeros((N,N),dtype = complex)
for a_op, nps in a_ops:
    A = ekets.conjugate()@a_op@ekets.T
    for a in range(N):
        # population outflow
        W[a,a] += -sum([A[a,b]*A[b,a]*nps(evals[a]-evals[b]) for b in range(N)])
        for b in range(N):
            # population inflow
            W[a,b] += A[b,a]*A[a,b]*nps(evals[b]-evals[a])
            
p0 = np.diag(ekets.conjugate()@rho0@ekets.T) # initial state
x = np.diag(ekets.conjugate()@X@ekets.T) # position operator

# propagation
t,p = 0,p0
times_p, pos_p = [], []
for dt in dts:
    WP = expm(W*dt) # Pauli propagator
    for m in range(1000):
        times_p.append(t) # append time 
        pos_p.append( np.real(x@p) ) # append position
        p = WP @ p # propagate state
        t += dt # propagate time
