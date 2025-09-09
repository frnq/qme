using LinearAlgebra
using GLMakie

σx = ComplexF64[0 1; 1 0]
σz = ComplexF64[1 0; 0 -1]
H = σz # Hamiltonian
Ls = [0.5 * σz, 0.2 * σx] # Lindblad operators
H_eff = H - im/2 * sum(L' * L for L in Ls) # effective Hamiltonian

ψ0 = ComplexF64[1, 1] / sqrt(2)
d = length(ψ0)  # dimension of the state

dt = 0.1 / norm(H_eff) # timestep
m = 200 # number of timesteps
t_final = dt * m # final time
times = range(0, t_final, length = m)

num_samples = 100 # number of samples
mean = zeros(ComplexF64, m) # array for the results
for count in 1:num_samples
    t = 0
    waves = [ψ0]

    for t in times[2:end]
        ψt = waves[end]
        u = rand() # random number in [0, 1)
        dps = [real(dt * (ψt' * (L'*L) * ψt)) for L in Ls]
        dP = sum(dps) # renormalization factor 1 - dP
        if dP < u
            temp = (I(d) - im * H_eff' * dt) * ψt
        else
            u = rand()
            Q = cumsum(dps) / dP
            k = searchsortedfirst(Q, u) # pick the jump that has occurred
            temp = Ls[k] * ψt
        end
        push!(waves, temp / norm(temp))
    end
    mean .+= [wave' * σx * wave for wave in waves]
end

mean ./= num_samples

fig, ax, l = lines(times, real(mean), label = "MCWF")
fig