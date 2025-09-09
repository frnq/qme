"""
    Liouvillian
    
∑[γ (L* L - 1/2 (1 ⊗ L†L + (L†L)ᵀ ⊗ 1))]
"""
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