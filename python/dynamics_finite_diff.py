def rho_dot(t,y):
    return P.dot(y)

# solution
t0,tf = 0,200
times = np.linspace(t0, tf, 50); 
sol = solve_ivp(rho_dot, [t0,tf], vec_rho_0, method = 'RK45', t_eval = times, vectorized = True)

fig, ax = plt.subplots()
ax.plot(ts*Omega, np.real(vec_rho_t[0]), 'k.-', label = r'$\rho_{00}(t)$' );
ax.plot(ts*Omega, np.real(vec_rho_t[3]), 'r.-', label = r'$\rho_{11}(t)$' );
ax.set_xlabel(r'time ($t\Omega$)')
ax.set_ylabel(r'population')
ax.legend();
ax.plot(times*Omega, np.real(sol['y'][0]), 'ko', 
        markersize = 10, alpha = 0.5, fillstyle='none', 
        label = r'$\rho_{00}(t)$ RK45' );
ax.plot(times*Omega, np.real(sol['y'][3]), 'ro', 
        markersize = 10, alpha = 0.5, fillstyle='none', 
        label = r'$\rho_{11}(t)$ RK45' );
ax.legend();
