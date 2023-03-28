function overlapprob = floquet_wave_vector_function(H0,Hint,omega,nph,measvec)

%H0 time independent Hamiltonian
%Hint interaction part of the Hamiltonian
%omega drive frequency
%nph number of photons (should be odd)
%measvec compute overlap of state with this vector
%H = H + Hint*cos(omega*t)

%Preallocate storage vectors for the absorption spectra and the
%eigenspectrum
overlapprob=0;

[evecs,~]=eigs(H0);

%atom
Hf=kron(eye(nph),H0);
%photons
maxn=floor(nph/2);
Hf=Hf+omega*kron(diag(-maxn:maxn),eye(2));
%interactions
tempv=zeros(1,nph);
tempv(2)=1;
Hf=Hf+kron(toeplitz(tempv),Hint);

tempv=zeros(1,nph);
tempv(maxn+1)=1;
psignd=kron(tempv,evecs(:,2)')';

[evecs_Hf,~]=eig(Hf);

%Sum over contributions from each of the photon manifolds, computing the
%overlap with the measurement vector for each of them.
for kc=1:nph
    tempv=zeros(1,nph);
    tempv(kc)=1;
    psim=kron(tempv,measvec)';
    for evc=1:length(Hf)
        overlap=(conj(psim')*evecs_Hf(:,evc))*(conj(evecs_Hf(:,evc)')*psignd);
        overlapprob=overlapprob+conj(overlap)*overlap;
    end
end

return
