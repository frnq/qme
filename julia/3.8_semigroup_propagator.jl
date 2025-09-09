using LinearAlgebra
using GLMakie
include("src/liouvillian.jl")

H = [0 1; 1 0.5]
c_ops = [[0 1; 0 0]]

superop = liouvillian(H, c_ops)
ρ0 = [1 0; 0 0]

dt = 0.5 # time step
m = 20 # number of time steps
d = size(ρ0, 1) # dimension of state
P = exp(superop * dt)
times_0, populations_0 = [], []
t = 0.0
ρ = ρ0

# propagate
for k in 1:m
    push!(populations_0, real(tr(ρ * ρ0)))
    push!(times_0, t)
    t += dt
    ρ = reshape(P * reshape(ρ, d^2), d, d)
end

function propagate(ρ0, superop, t)
    ρt_vec = exp(superop * t) * reshape(ρ0, d^2)
    return reshape(ρt_vec, d, d)
end

# time steps
times_1 = range(0, 10, 100)
# Population dynamics of the initial state
populations_1 = [real(tr(propagate(ρ0, superop, t) * ρ0)) for t in times_1]

# Plot
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "t", ylabel = "Tr[ρ(t)ρ(0)]")
scatter!(ax, times_0, populations_0, color = :black, label = "w/ semigroup")
lines!(ax, times_1, populations_1, color = :blue, label = "w/ matrix exp")

axislegend(ax)
fig