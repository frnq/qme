using LinearAlgebra

d = 3
basis = I(d)
ψs = [basis[:, j] for j in 1:d] # basis states
ps = [0.1, 0.3, 0.6]  # probabilities (normalized)

# Mixed state density matrix from outer products
# The ' symbol performs the conjugate transpose
ρ = sum(p * ψ * ψ' for (ψ, p) in zip(ψs, ps))
A = Diagonal(1:3)  # some operator

# Expectation value
exp_A = tr(ρ * A)
