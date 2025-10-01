using LinearAlgebra
include("4.3_random_quantum_network.jl")

times, positions = simulate_dynamics(dts, ρ_vec, X_vec)

W = zeros(N, N)

for (a_op, nps) in a_ops
    A = kets' * a_op * kets
    for a in 1:N
        W[a, a] -= sum([A[a, b] * A[b, a] * nps(vals[a] - vals[b]) for b in 1:N])
        for b in 1:N
            W[a, b] += A[b, a] * A[a, b] * nps(vals[b] - vals[a])
        end
    end
end

p0 = diag(kets' * ρ * kets) # initial state
x = diag(kets' * X * kets) # position operator

t = 0
p = p0
times_p = []
positions_p = []

for dt in dts
    WP = exp(W * dt)
    for m in 1:1000
        push!(times_p, t)
        push!(positions_p, real(dot(x, p)))
        p = WP * p
        t += dt
    end
end
times_p
positions_p

fig = Figure()
ax = Axis(fig[1, 1],
    title = "Random Quantum Network Exciton Dynamics",
    xlabel = "Time (fs)",
    ylabel = "Position (a.u.)",
    xscale = log10,
)
lines!(times, positions, label = "Bloch-Redfield ME")
lines!(times_p, positions_p, linestyle = :dash, label = "Pauli ME")

xlims!(1, 1e6)
axislegend(ax)

fig