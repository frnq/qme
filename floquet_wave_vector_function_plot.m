%Simple example of using Floquet theory (ala Shirley) to numerical 
%integrate the evolution of a two-level system and compute the absorption
%spectrum
%
%script to generate a demonstration plot.
%The floquet function can be easily vectorised, but for simplicity here it
%is called seperately for each value of the Hamiltonian

%Hamiltonian here is H = H0 + Hint*cos(omega*t)
%where H0=0.5*Delta*sigma_z + epsilon*sigma_x
%and Hint=0.5*Vstr*sigma_z;

%Define the absorption as the probability of measuring in the state measvec
measvec=[1,0];

%Example values of the Hamiltonian parameters
epsilon=0.2;
%Range of delta to consider
Delta_range=-6:0.02:6;
%Drive frequency
omega=1.5;

%number of photon manifolds (should be odd)
nph=13;

%number of different drive strengths to consider
Vstr_range = [0.05, 0.2, 1];

%Preallocate storage vectors for the absorption spectra and the
%eigenspectrum
Absorp_av=zeros(length(Delta_range),length(Vstr_range));
spec=zeros(2,length(Delta_range));

for vc = 1:length(Vstr_range)
    Vstr = Vstr_range(vc);

    for jc=1:length(Delta_range)
        delta=Delta_range(jc);

        H0=[delta/2,epsilon;epsilon,-delta/2];
        [evecs,evs]=eigs(real(H0));

        spec(:,jc)=diag(evs);
        Hint=[1,0;0,-1]/2;

        Absorp_av(jc,vc) = floquet_wave_vector_function(H0,Vstr*Hint,omega,nph,measvec);
    end

end

%figure;
subplot(2,1,1);
hp = plot(Delta_range,spec,'linewidth',1);
xlabel('detuning (\delta)');
ylabel('Energy');
hold on
plot(-3*[omega omega],[-4 4],'k--');
plot(-2*[omega omega],[-4 4],'k--');
plot(-1*[omega omega],[-4 4],'k--');
plot(1*[omega omega],[-4 4],'k--');
plot(2*[omega omega],[-4 4],'k--');
plot(3*[omega omega],[-4 4],'k--');
hold off

subplot(2,1,2);
plot(Delta_range,Absorp_av,'linewidth',1);
ylabel('Absorption probability');
xlabel('detuning (\delta)');
legend('Vstr = 0.05', 'Vstr = 0.20', 'Vstr = 1.00','location','northwest');

