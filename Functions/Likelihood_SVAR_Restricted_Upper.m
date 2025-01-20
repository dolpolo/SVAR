function [logLik]=Likelihood_SVAR_Restricted_Upper(teta)

global Sigma_1Regime
global Sigma_2Regime
global Sigma_3Regime
global Sigma_4Regime

global T1
global T2
global T3
global T4

C=[teta(1) teta(3) 0;
   teta(2) teta(4) 0;
   0       0       teta(5)];

Q2=[teta(6) 0       teta(10);
    teta(7) teta(9) 0;
    teta(8) 0       teta(11)];

Q3=[0        0        teta(14);
    teta(12) teta(13) teta(15);
    0        0        teta(16)];

Q4=[teta(17)    0      teta(20);
    teta(18)  teta(19) teta(21);
    0           0      teta(22)];

    K1 = (C);
    K1 = K1^(-1);
    K2 = (C+Q2);
    K2 = K2^(-1);
    K3 = (C+Q2+Q3);
    K3 = K3^(-1);
    K4 = (C+Q2+Q3+Q4);
    K4 = K4^(-1);
    
    T=T1+T2+T3+T4;
    M=size(C,1);

    logLik=-(-0.5*T*M*(log(2*pi))...
        +0.5*T1*log((det(K1))^2)-0.5*T1*trace(K1'*K1*Sigma_1Regime)...
        +0.5*T2*log((det(K2))^2)-0.5*T2*trace(K2'*K2*Sigma_2Regime)...
        +0.5*T3*log((det(K3))^2)-0.5*T3*trace(K3'*K3*Sigma_3Regime)...   
        +0.5*T4*log((det(K4))^2)-0.5*T4*trace(K4'*K4*Sigma_4Regime));

end