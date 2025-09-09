# These are just some basic checks and not a full test suite
# The Project.toml for the Julia code does not include
# a name or UUID so `runtests` from the REPL will not work.

using LinearAlgebra
using Test
using ToeplitzMatrices
include("../src/functions.jl")

@testset "Toeplitz" begin
    a = [0, 1, 0]
    T = Toeplitz(a, a)
    off_diagonals = Tridiagonal(ones(length(a)-1), zeros(length(a)), ones(length(a)-1))
    @test T == [0 1 0
                1 0 1
                0 1 0]

    @test T == off_diagonals
end

@testset "onehot" begin
    n = 5
    for k in 1:n
        v = onehot(n, k)
        @test v == [i == k for i in 1:n]
    end
end

@testset "annihilation" begin
    n = 5
    a = annihilation(n)
    @test a == [0 sqrt(1) 0 0 0 0
                0 0 sqrt(2) 0 0 0
                0 0 0 sqrt(3) 0 0
                0 0 0 0 sqrt(4) 0
                0 0 0 0 0 sqrt(5)
                0 0 0 0 0 0]
    @test a' * a ≈ Diagonal(0:5)
end