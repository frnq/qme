# construct W tensor
W = np.zeros((N,N),dtype = complex)
for a_op, nps in a_ops:
    A = a_op.transform(ekets)
    for a in range(N):
        # population outflow
        W[a,a] += -sum([A[a,b]*A[b,a]*nps(evals[a]-evals[b]) for b in range(N)])
        for b in range(N):
            # population inflow
            W[a,b] += A[b,a]*A[a,b]*nps(evals[b]-evals[a])
            
p0 = np.diag(rho0.transform(ekets).full()) # initial state
x = np.diag(X.transform(ekets).full()) # position operator

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
# plot
plt.plot(np.array(times)/fs, np.array(pos), 'k-', label = 'Full BR ME')
plt.plot(np.array(times_p)/fs, np.array(pos_p), 'r--', label = 'Pauli ME')
plt.xscale('log')