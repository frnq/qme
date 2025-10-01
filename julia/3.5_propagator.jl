using LinearAlgebra
using GLMakie
include("src/liouvillian.jl")

H = [0 1; 1 0.5]
c_ops = [[0 1; 0 0]]
ρ0 = [1 0; 0 0]

superop = liouvillian(H, c_ops)

function propagate(ρ0, superop, t)
    d = size(ρ0, 1) # dimension of the system
    propagator = exp(superop * t)  # matrix exponentiation is built into the default exp
    ρ_vec = reshape(ρ0, d^2)
    ρ_vec_t = propagator * ρ_vec
    reshape(ρ_vec_t, d, d)  # return ρ(t)
end

# time steps
times = range(0, 10, length = 100)
# Population in ρ0 in time
pop = [real(tr(propagate(ρ0, superop, t) * ρ0)) for t in times]

fig, ax, l = lines(times, pop)
ax.xlabel = "t"
ax.ylabel = "Tr[ρ(t)ρ₀]"
fig