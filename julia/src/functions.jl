

"""
    annihilation(n)

The annihilation operator for a Hilbert space of dimension `n`.
"""
function annihilation(n_max)
    d_photon = n_max + 1
    a = zeros(Float64, d_photon, d_photon)
    for n in 1:n_max
        a[n, n + 1] = sqrt(n)
    end
    a
end

"""
    onehot(n, k)

Create a one-hot encoded vector of length `n` with a 1 at position `k`.
"""
function onehot(n, k)
    v = zeros(Int, n)
    v[k] = 1
    v
end