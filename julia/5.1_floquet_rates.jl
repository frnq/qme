using LinearAlgebra

# The following functions are defined in src/functions.jl
# - annihilation(n)
# - onehot(n, k)
include("src/functions.jl")

function floquet(H_0, H_int, ω, n_ph, meas_vec)
    dim = size(H_0, 1)
    n_max = floor(Int, n_ph / 2)

    H_atom = kron(I(n_ph), H_0) # atomic Hamiltonian
    H_ph = ω * kron(Diagonal(-n_max:n_max), I(dim)) # photon Hamiltonian

    # Construct the interaction Hamiltonian
    # by building an ad-hoc Toeplitz matrix
    off_diagonals = Tridiagonal(ones(n_ph-1), zeros(n_ph), ones(n_ph-1))
    H_int = kron(off_diagonals, H_int)
    
    H = H_atom + H_int + H_ph # Put it all together

    # spectral decomposition of H0
    vecs_0 = eigvecs(H_0)
    vecs = eigvecs(H)

    ψ_g = onehot(n_ph, n_ph)  # ground state
    ψ_g = kron(ψ_g, vecs_0[1, :])

    overlap_probability = 0
    for k_c in 1:n_ph
        k_vec = onehot(n_ph, k_c)
        ψ_m = kron(k_vec, meas_vec)
        for vec in eachcol(vecs)
            overlap_probability += abs2((ψ_m' * vec) * (vec' * ψ_g))
        end
    end
    overlap_probability
end