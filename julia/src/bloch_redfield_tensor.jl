δ(x, y) = x == y ? 1 : 0

"""
    Noise power spectrum
"""
function S(ω, ωc, η, β, threshold=1e-10)
    if ω > threshold || ω <= -threshold
        return 2π * η * ω * exp(-abs(ω) / ωc) / (1 - exp(-ω * β) + threshold)
    elseif -threshold < ω < threshold
        return 2π * η / β
    else
        return 0
    end
end

"""
    Bloch-Redfield tensor
"""
function BR_tensor(H, a_ops, secular=true, secular_cutoff=0.01)
    dim = size(H, 1) # system dimension
    λs, ekets = eigen(H) # These are already sorted, see documentation for details
    a_ops_S = [[ekets' * A * ekets, nps] for (A, nps) in a_ops] # transform a_ops to Hamiltonian basis
    indices = [(a, b) for a in 1:dim for b in 1:dim]
    
    #Bohr frequencies
    BohrF = sort([λs[a] - λs[b] for a in 1:dim for b in 1:dim]) # sorted list of Bohr frequencies

    R = zeros(ComplexF64, dim^2, dim^2)
    for (j, (a, b)) in enumerate(indices)
        for (k, (c, d)) in enumerate(indices)
            R[j, k] += -im * (λs[a] - λs[b]) * (a == c && b == d)
            for (a_op, nps) in a_ops_S
                gmax = maximum([NPS(f) for f in BohrF]) # largest rate for secular approximation
                A = a_op # coupling operator
                # secular approximation test
                if secular == true && abs(λs[a] - λs[b] - λs[c] + λs[d]) > gmax * secular_cutoff
                    continue
                else
                    # non-unitary part
                    R[j, k] += -0.5 * (
                                δ(b, d) * sum(A[a, n] * A[n, c] * nps(λs[c] - λs[n]) for n in 1:dim) -
                                A[a, c] * A[d, b] * nps(λs[c] - λs[a]) +
                                δ(a, c) * sum(A[d, n] * A[n, b] * nps(λs[d] - λs[n]) for n in 1:dim) -
                                A[a, c] * A[d, b] * nps(λs[d] - λs[b])
                    )
                end
            end
        end
    end
    return R
end