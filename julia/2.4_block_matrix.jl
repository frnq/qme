using LinearAlgebra

# LinearAlgebra is only needed for I(n).
d_a, d_b = 2, 3 # dimensions of the bases
basis = [I(d_a) zeros(d_a, d_b); zeros(d_b, d_a) I(d_b)] # block matrix