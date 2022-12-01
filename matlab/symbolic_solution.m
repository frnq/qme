syms Omega Delta Gamma hbar real    
rho = sym('rho', [2 2])     % Displays rho = [rho1_1 rho1_2; rho2_1 rho2_2] 
assume(trace(rho)==1);      % Imposing an assumption 

% Obtaining the simplified master equation for Omega=0
H_s = [0 hbar*Omega; hbar*Omega hbar*Delta]; 
L = [0 1; 0 0];     
rho_dot = simplify((-1i/hbar)*(H_s*rho - rho*H_s) ...
    + Gamma*(L*rho*L' - (1/2)*(L'*L*rho + rho*(L')*L)));
rho_dot = subs(rho_dot, Omega, 0);

% Solving for steady state rho and arranging solution in matrix form
S = solve(rho_dot==0, rho);         
rho_inf = [S.rho1_1 S.rho1_2; S.rho2_1 S.rho2_2] % Displays rho_inf = [1 0;0 0]
