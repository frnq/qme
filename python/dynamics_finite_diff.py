from scipy.integrate import solve_ivp

def rho_dot(t,y):
    return superop.dot(y)

# solution
t0,tf = 0,200
times = np.linspace(t0, tf, 20); 
sol = solve_ivp(rho_dot, [t0,tf], vec_rho_0, method = 'RK45', t_eval = times, vectorized = True)

fig, ax = plt.subplots(figsize=(6,2))
ax.plot(ts*Omega, np.real(vec_rho_t[3]), 'b-', label = r'SVD' );
ax.set_xlabel(r'time ($t\Omega$)', usetex= True, fontsize = 10);
ax.set_ylabel(r'$\mathrm{Tr}[\rho(t)\rho_0]$', usetex= True, fontsize = 10);
ax.legend();
ax.plot(times*Omega, np.real(sol['y'][3]), 'ks', markersize = 4,alpha = 1,
        label = r'RK45' );
ax.legend();
