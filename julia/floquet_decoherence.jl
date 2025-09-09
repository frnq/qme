using LinearAlgebra
include("src/liouvillian.jl")
# Function to obtain the Floquet propagator U(t) in terms of Floquet blocks
# U -> U(t) = ( ..., U[2], U[1], U[0], U[-1], U[-2], ...)
# for a harmonically driven system
# arguments
# times:    time steps
# omega:    driving frequency
# nphotons: number of photons driving the interaction
# P0:       time-independent part of the Liouville superoperator
# Pint:     superoperator of the interaction
# average:  if average is True, average over all possible phases of driving field.

function floquet_decoherence(times, ω, n_photons, P0, P_int, average=false)
    if n_photons % 2 == 0
        error("n_photons must be odd")
    end
    d = size(P0, 1)
    U0 = I(d) # initial generator

    n_max = floor(Int, n_photons / 2)
    H_ph = ω * kron(Diagonal(-n_max:n_max), I(d))

    # system
    off_diagonals = Tridiagonal(ones(n_photons-1), zeros(n_photons), ones(n_photons-1))
    Pf = kron(I(n_photons), P0) + H_ph + kron(off_diagonals, P_int)

    vals, vecs = eigen(Pf)

    U0_f = kron(onehot(n_photons, n_max+1)', U0)
    U_t = zeros(ComplexF64, d, d, length(times))

    for i in eachindex(times)
        U_eigs = Diagonal(exp.(-im * vals * times[i]))
        U_f = vecs * U_eigs * inv(vecs) * U0_f'

        if average == true
            U_t[:, :, i] += U_f[(1:d) .+ n_max * d, :]
        else
            for j in -n_max:n_max
                counter = j + n_max
                U_t[:, :, i] += U_f[(1:d) .+ counter * d, :] * exp(im * j * ω * times[i])
            end
        end
    end
    U_t
end


# meas_vec = [1, 0]
# ϵ = 0.2
# Δ = 1.0
# ω = 1.5
# n_ph = 3
# Γ = 0.1
# L = [1 0; 0 -1]
# Ls = [sqrt(Γ) * L]
# H0 = Δ/2 * L + ϵ * [0 1; 1 0]
# H_int = [1 0; 0 -1] / 2  # interaction Hamiltonian
# P0 = liouvillian(H0, Ls)
# P_int = liouvillian(H_int, [])

# floquet_decoherence([10/Γ], ω, n_ph, P0, P_int)