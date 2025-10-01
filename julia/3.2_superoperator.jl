using LinearAlgebra

# constructs the Liouville superoperator from
# the Hamiltonian and
# the set of Lindblad operators rescaled by the root of the rates

# ∑[γ (L* L - 1/2 (1 ⊗ L†L + (L†L)ᵀ ⊗ 1))]
function liouvillian(H, Ls, ħ = 1)
    d = size(H, 1)
    superH = -im/ħ * (kron(I(d), H) - kron(transpose(H), I(d)))

    if isempty(Ls) # if there are no Lindblad operators (necessary for script 3.7)
        superL = zeros(ComplexF64, d^2, d^2)
    else
        superL = sum(L -> kron(conj(L), L) - 0.5 * (kron(I(d), L'*L) + kron(transpose(L'*L), I(d))), Ls)
    end
    
    superH + superL
end

H = [0 1 ; 1 1] # some Hamiltonian
Ls = [[0 1 ; 0 0]] # array with a single Lindblad operator with embedded rate
superop = liouvillian(H, Ls) # Liouville superoperator

# The Liouvillian is used repeatedly
# in the following examples, so we have
# reproduced this function in src/liouvillian.jl