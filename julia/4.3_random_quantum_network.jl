using LinearAlgebra
using GLMakie
using Random

include("src/bloch_redfield_tensor.jl")

# constants and units
fs = 1 / (2.4189e-2) # femtosecond in Hartree AU
eV = 1 / (fs * 0.6582) # electronvolt in Hartree AU

Random.seed!(0) # fix random seed
N = 10 # number of sites

# disorder parameters
σ_E = 100e-3 * eV
σ_V = 50e-3 * eV

# random energies
Es = σ_E * randn(N)

H = Matrix(Diagonal(Es)) # Hamiltonian

# random couplings
pairs = [(j, k) for k in 2:N for j in 1:k-1]
sel = randperm(length(pairs))[1:N]

for idx in sel
    j, k = pairs[idx]
    v = σ_V * randn()
    H[j, k] = v
    H[k, j] = conj(v) # Hermiticity
end

vals, kets = eigen(Hermitian(H))

NPS(ω) = S(ω, 150e-3 * eV, 1e-1, 1/(25e-3 * eV))

# site projection operators
proj(k, N) = Matrix(Diagonal([i == k ? 1.0 : 0.0 for i in 1:N]))
a_ops = [(proj(k, N), NPS) for k in 1:N]

R = BR_tensor(H, a_ops)

# position operator
X = Diagonal(1:N)
X_vec = reshape(kets' * X * kets, N^2)
ρ = zeros(N, N)
ρ[1, 1] = 1.0 # initial state
ρ_vec = reshape(kets' * ρ * kets, N^2)

dts = [1e-1, 1e1, 1e3] * fs

function simulate_dynamics(dts, ρ_vec, X_vec)
    t = 0
    times = []
    positions = []
    for dt in dts
        P = exp(R * dt)
        for m in 1:1000
            if t > 0
                push!(times, t)
                push!(positions, real(ρ_vec' * X_vec))
            end
            ρ_vec = P * ρ_vec
            t += dt
        end
    end
    return times, positions
end