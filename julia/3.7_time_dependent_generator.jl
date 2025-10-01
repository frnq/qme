using LinearAlgebra
using GLMakie
include("src/liouvillian.jl")

# system parameters
ω0 = 2
ω = 3
γ = 0.3

σz = [1 0; 0 -1]
σx = [0 1; 1 0]
H0 = ω0 * σz / 2  # Hamiltonian H0
H1 = ω0 * σx / 2  # Hamiltonian H1

c_ops = [sqrt(γ) * [0 0; 0 1]]  # Lindblad operator
L0 = liouvillian(H0, c_ops)  # superoperator L0
L1 = liouvillian(H1, [])  # superoperator L1 without time dependence
ρ0 = [1 0; 0 0]  # initial state
d = size(ρ0, 1)  # dimension of the system

v(t) = cos(ω * t)  # time-dependent coupling

function dynamics(time_final, n_samples)
    times = range(0, stop=time_final, length=n_samples) # time steps
    dt = times[2] - times[1]  # finite difference
    ρt_vec = reshape(ρ0, d^2)  # initial state vectorized
    
    populations = []
    # propagation
    for t in times
        ρt = reshape(ρt_vec, d, d)  # reshape to density matrix
        push!(populations, real(tr(ρt * ρ0))) # append to populations
        superop = L0 + v(t) * L1  # update superoperator
        P = exp(superop * dt) # propagator (exp() can operate on matrices in Julia)
        ρt_vec = P * ρt_vec  # propagate state for dt
    end
    return times, populations
end

# results
t_final = 30
times = range(0, t_final, length=200)

data_hi = dynamics(t_final, 1000)  # high-resolution data
data_mid = dynamics(t_final, 50)
data_low = dynamics(t_final, 10)

# Plot
fig = Figure()
ax1 = Axis(fig[1, 1], xlabel="t", ylabel="Tr[ρ(t)ρ₀]")

# The splat operation "..." unpacks the tuple into separate arguments
lines!(data_hi..., label = "sample = $(length(data_hi[1]))")
scatterlines!(data_mid..., color = (:orangered, 0.7), linestyle = :dash, label = "sample = $(length(data_mid[1]))")
scatterlines!(data_low..., color = (:green, 0.5), linestyle = :dash, label = "sample = $(length(data_low[1]))")

ax2 = Axis(fig[2, 1], xlabel="t", ylabel="v(t)")
lines!(times, v.(times))

rowsize!(fig.layout, 2, 100) # fix row 2 height to 150
axislegend(ax1)
hidexdecorations!(ax1, grid=false)
fig