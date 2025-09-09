using LinearAlgebra

function is_state(ρ)
    vals = eigvals(ρ)
    non_unit = 1 - tr(ρ)
    non_hermitian = norm(ρ - ρ')
    non_positive =  sum((abs.(vals) .- vals) / 2)

    return 1 - norm([non_unit, non_hermitian, non_positive])
end

# a state
ρ_1 = [0.2 0 0
       0 0.3 0
       0 0 0.5]
is_state(ρ_1) # should return 1

# a state with some error
ρ_2 = ρ_1 + [-1e-4 1e-6 0; 0 0 1e-2im; 1e-3im 0 1e-4]
is_state(ρ_2) # should return something close to 1

# Note: eigvals is evaluating very small imaginary eigenvalues to zero in this example.
