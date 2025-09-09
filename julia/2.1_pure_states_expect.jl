using LinearAlgebra

ψ = [1, im, -1] / sqrt(3)
A = Diagonal([1, 2, 3]) # some operator
# expectation value of A in state ψ
exp_A = real(dot(ψ, A * ψ)) # dot conjugates the first argument