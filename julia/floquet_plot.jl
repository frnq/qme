using GLMakie
include("5.1_floquet_rates.jl")


σz = [1 0; 0 -1]
σx = [0 1; 1 0]

ω = 1.5
ϵ = 0.2
n_ph = 13
meas_vec = [1, 0]  # measure in excited state
Vs = [0.05, 0.2, 1.0]
Δs = range(-6, 6, length = 600)

spectrum = zeros(2, length(Δs))
abs_av = zeros(length(Δs), length(Vs))

for (i, Δ) in enumerate(Δs)
    H_0 = Δ/2 * σz + ϵ * σx
    vals, vecs = eigen(H_0)
    spectrum[:, i] = vals

    for (j, V) in enumerate(Vs)
        H_int = V * σz / 2
        abs_av[i, j] = floquet(H_0, H_int, ω, n_ph, meas_vec)
    end
end


fig = Figure()
ax1 = Axis(fig[1, 1],
    ylabel = "Energy (λ)",
    title = "Floquet Spectrum",
    xticks = LinearTicks(5))
lines!(Δs / ω, spectrum[2, :], label = "Excited State")
lines!(Δs / ω, spectrum[1, :], label = "Ground State")

ax2 = Axis(fig[2, 1],
    xlabel = "Detuning Δ/ω",
    ylabel = "Absorption Probability",
    xticks = LinearTicks(5))
lines!(-Δs / ω, abs_av[:, 1], label = "V = $(Vs[1])")
lines!(-Δs / ω, abs_av[:, 2], label = "V = $(Vs[2])")
lines!(-Δs / ω, abs_av[:, 3], label = "V = $(Vs[3])")

xlims!(ax1, -4, 4)
xlims!(ax2, -4, 4)
axislegend(ax1, position = :rc)
axislegend(ax2, position = :lt)
fig