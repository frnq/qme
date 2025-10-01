# Does not work yet

using ModelingToolkit
using LinearAlgebra

@variables a b Γ Ω Δ

ρ = [a b; conj(b) 1 - a]

H = [0 Ω
     Ω Δ]

L = Num[0 1
     0 0]

ρ̇ = -im*(H*ρ - ρ*H) + Γ*(L*ρ*L' - 0.5*(L'*L*ρ + ρ*(L'*L)))

eqs_matrix = substitute(ρ̇, Dict(Ω => 0))
eqs = vec(eqs_matrix) .~ 0

# Solve the system
@variables t
ns = NonlinearSystem(eqs, [a, b], [])
sol = solve(ns)