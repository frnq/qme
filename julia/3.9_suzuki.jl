using LinearAlgebra
using GLMakie
include("src/liouvillian.jl")

σx = [0 1; 1 0]
σy = [0 -im; im 0]
σz = [1 0; 0 -1]
σp = σx + im * σy / 2
σm = σx - im * σy / 2

superop_1 = liouvillian(σx, [σp])
superop_2 = liouvillian(σy, [σm])
ρ0 = [1 0; 0 0]
d = size(ρ0, 1) # dimension of state and superoperator
ms = [50, 100, 1000]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "t", ylabel = "Tr[ρ(t)ρ(0)]")

for (km, m) in enumerate(ms)
    dt = 10/m
    P_1 = exp(superop_1 * dt)
    P_2 = exp(superop_2 * dt)
    P = P_2 * P_1 # Suzuki-Trotter expansion for small times dt
    times_0 = []
    populations_0 = []
    t = 0.0
    ρ = ρ0

    for k in 1:m
        push!(populations_0, tr(ρ * ρ0))
        push!(times_0, t)
        t += dt
        ρ = reshape(P * reshape(ρ, d^2), d, d)
    end

    lines!(ax, times_0, real(populations_0), label = "m = $m")
end

axislegend(ax)
fig