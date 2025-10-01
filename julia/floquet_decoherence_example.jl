using LinearAlgebra
include("src/functions.jl")
include("src/liouvillian.jl")
include("floquet_decoherence.jl")

# H = H0 + H_int * cos(ω * t)  Total Hamiltonian
# H0 = Δ * σz / 2 + ϵ * σx  Bare Hamiltonian
# H_int = V * σz / 2  Interaction Hamiltonian
# L = Γz * σz  Decoherence (dephasing)

# Define the absorption as the probability of measuring in the excited state
meas_vec = [1, 0]

# Example values of the Hamiltonian parameters
ϵ = 0.2             # energy
Δs = -6.0:0.02:6.02 # detuning range
ω = 1.5             # driving frequency
n_ph = 13           # number of photons

# Dephasing operator and rate
L = [1 0; 0 -1]
Γzs = [0.005, 0.05, 0.1]
A_op = meas_vec * meas_vec'

absorb_av = zeros(length(Δs), length(Γzs))
spectrum = zeros(2, length(Δs))

for (i, Γz) in enumerate(Γzs)
    Ls = [sqrt(Γz) * L]

    for (j, Δ) in enumerate(Δs)
        H_0 = Δ/2 * L + ϵ * [0 1; 1 0]
        vals, vecs = eigen(H_0)
        spectrum[:, j] = vals

        ρ0 = vecs[:, 1] * vecs[:, 1]' # start in ground state
        ρ0_vec = reshape(ρ0, :, 1)

        H_int = L / 2  # interaction Hamiltonian
        system_1 = I(length(H_0))
        P0 = liouvillian(H_int, [])

        times = [10 / Γz]

        U_sum = floquet_decoherence(times, ω, n_ph, P0, P_int)
        ρ_t = reshape(U_sum[:, :, 1] * ρ0_vec, 2, 2)
        absorb_av[j, i] = real(tr(A_op * ρ_t))
    end
end

fig = Figure()
ax1 = Axis(fig[1, 1],
    ylabel = "Energy")
lines!(Δs / ω, spectrum[1, :])
lines!(Δs / ω, spectrum[2, :])

ax2 = Axis(fig[2, 1])
for i in eachindex(Γzs)
    lines!(Δs / ω, -absorb_av[:, i], label = "Γz = $(Γzs[i])")
end

fig