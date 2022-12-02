eqns = [rho_dot(1,1)==0, rho_dot(1,2)==0, rho_dot(2,1)==0, rho_dot(2,2)==0];
S = solve(eqns, rho);         
rho_inf = [S.rho1_1 S.rho1_2; S.rho2_1 S.rho2_2] % Displays rho_inf = [1 0;0 0]
