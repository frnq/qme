using LinearAlgebra

ψ = [1, 2im, 0, -2, 0] # some (non-normalized) state
ρ = ψ * ψ'
ρ /= tr(ρ)
d = length(ψ)
ρ_vec = reshape(ρ, d^2, 1) # vectorized density matrix