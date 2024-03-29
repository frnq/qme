R = [R_mat vec_rho_0];          % Concatenating R_mat and vec_rho_0
R = rref(R);                    % Converting R to reduced row echcelon form

vec_rho_t2 = zeros(length(vec_rho_0), length(t)); 	
for k=1:(d^2)
    b(k) = R(k,5);              % kth coefficient                    
    vec_rho_t2(:,:) = vec_rho_t2(:,:) + b(k)*R_mat(:,k)*exp(D(k,k)*t);
end
