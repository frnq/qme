using LinearAlgebra
using GLMakie
include("src/liouvillian.jl")

# Parameters
Ω = 0.05
Γ = Ω / 5

# Build the superoperator
H_s = [0 Ω
       Ω 0]
d = size(H_s, 1)
L = sqrt(Γ) * [0 1; 1 0]
superop = liouvillian(H_s, [L])

# Right eigenvectors: superop * v = λ * v
λ, right = eigen(superop)
left = pinv(right')

# Initial state in vectorized form
ρ0_vec = [0, 0, 0, 1]

# Solution
ts = range(0, 200, length = 200)
ρt_vec = zeros(length(ρ0_vec), length(ts))

for k in 1:d^2
    norm = left[:, k]' * right[:, k]
    ρt_vec += (left[:, k]' * ρ0_vec * right[:, k]) * exp.(λ[k] * ts)' / norm
end

# Population dynamics
fig = Figure(size = (800, 400))
ax = Axis(fig[1, 1], xlabel = "time (tΩ)", ylabel = "Tr[ρ(t)ρ₀]")
lines!(ax, ts * Ω, real(ρt_vec[4, :]), label = "k-")
fig