using LinearAlgebra

# Include file containing the Liouvillian function
include("src/liouvillian.jl")

# two-level Hamiltonian from delta and omega parameters
H_tls(ω, Δ) = [0 ω; ω Δ]

# Lindblad operators for spontaneous relaxation and dephsing
L_relax(γ) = γ * [0 1; 0 0]
L_dephase(γ) = γ * [0 0; 0 1]

# ------- (i) relaxation with driving and no dephasing --------
ω = 1
Δ = 1
γ_relax = 1
γ_dephase = 0

superop = liouvillian(H_tls(ω, Δ), [L_relax(γ_relax), L_dephase(γ_dephase)])
null = nullspace(superop)
println("(i) The steady-state space is a linear subpace of dimension = +$(string(length(transpose(null))-1))")
ρ_ss = reshape(null, 2, 2)
ρ_ss /= tr(ρ_ss)


# ------- (ii) dephasing with no driving --------
ω = 0
Δ = 1
γ_relax = 0
γ_dephase = 1

superop = liouvillian(H_tls(ω, Δ), [L_relax(γ_relax), L_dephase(γ_dephase)])
null = nullspace(superop)
println("(ii) The steady-state space is a linear subpace of dimension = +$(string(length(transpose(null))-1))")