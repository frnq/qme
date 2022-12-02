[t_ode, vec_rho] = ode45(@(t_ode, vec_rho) odefun(t_ode,...
    vec_rho, H_s, L, Gamma, hbar), t, vec_rho_0);

% Define the following function as a new file
function vec_rho_dot = odefun(t_ode, vec_rho, H_s, L, Gamma, hbar)
rho = reshape(vec_rho, 2, 2);                       % Reshaping to matrix form
rho_dot = (-1i/hbar)*(H_s*rho - rho*H_s) +...
    Gamma*(L*rho*L' - (1/2)*(L'*L*rho + rho*L'*L)); % Master equation
vec_rho_dot = reshape(rho_dot, 4, 1);               % Reshaping to vector form
end
