hbar = 1;   d=2;   Omega = 0.05;   Gamma = Omega/5;
t = linspace(0, 200, 50); 

% Building the superoperator
H_s = [0 hbar*Omega; hbar*Omega 0];
L = [0 1; 1 0];
I = eye(2);
P = (-1i/hbar)*(kron(I, H_s) - kron(H_s.', I)) ...
    + Gamma*(kron(conj(L), L) - (1/2)*(kron((L'*L).',I) + kron(I, L'*L)));

% Finding matrices containing right and left eigenvectors:
% D is a diagonal matrix of eigenvalues
% Columns of R_mat and L_mat correspond to left and right eigenvectors s.t.
% P*R_mat = R_mat*D and L_mat'*P = D*L_mat' (where ' denotes conjugate transpose)
[R_mat, D, L_mat] = eig(P);

vec_rho_0 = [0 0 0 1]';                              % Initial rho vector
vec_rho_t = zeros(length(vec_rho_0), length(t));    % Initializing solution

for k=1:(d^2)
    % Obtaining normalized eigenvectors and coefficients
    norm_fac = sqrt(L_mat(:,k)'*R_mat(:,k));
    L_k_dag_norm = L_mat(:,k)'/norm_fac;
    R_k_norm = R_mat(:,k)/norm_fac;
    a(k) = L_k_dag_norm*vec_rho_0;                  
    
    % Summing the kth solution 
    vec_rho_t(:,:) = vec_rho_t(:,:) + a(k)*R_k_norm*exp(D(k,k)*t);
end

% Real part plotted, ignoring small imaginary errors of numerical eigensolutions
plot(t, real(vec_rho_t(4,:)), 'ro'); hold on
