syms Omega Delta Gamma hbar real
% Obtaining the superoperator symbolically
H_s = [0 hbar*Omega; hbar*Omega hbar*Delta]; 
L = [0 1; 0 0];     
I = [1 0; 0 1];
P = (-1i/hbar)*(kron(I,H_s) - kron(H_s', I)) + Gamma*(kron(conj(L),L) -...
    (1/2)*(kron((L'*L)', I) + kron(I, L'*L)));

vec_rho_1 = null(subs(P, Omega, 0)) % Solving for Omega = 0
vec_rho_2 = null(subs(P, Gamma, 0)) % Solving for Gamma = 0
