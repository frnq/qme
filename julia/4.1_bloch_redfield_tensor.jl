using LinearAlgebra

# The Bloch-Redfield tensor and noise power spectrum
# functions are calculated in the src directory
# since they are used in multiple examples

include("src/bloch_redfield_tensor.jl")

NPS(ω) = S(ω, 1, 1, 2)

# spin parameters
ϵ0 = 1
Δ = 0.2
σz = [1 0; 0 -1]
σx = [0 1; 1 0]
HS = ϵ0 * σz / 2 + Δ * σx / 2 # spin Hamiltonian
a_ops = [[σz, NPS]] # list of tuples of coupling operator and associated NPS

# compute the Bloch-Redfield tensor in the Hamiltonian basis
BR_tensor(HS, a_ops)