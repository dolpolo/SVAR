function [logLik]=Likelihood_UNRESTRICTED(A)

global NLags
global VAR_Variables_X
global VAR_Variables_Y
global SigmaProva

ExogenousRegressos = 0;

[T, M]= size(VAR_Variables_Y);

%% NoTrend
% A_VAR_COMP=[A(:,2:end);
%     eye(M*(NLags-1)) zeros(M*(NLags-1),M)];
%     if max(eig(A_VAR_COMP)>=1)
%     logLik=100000;
%     else
%     Error=VAR_Variables_Y-VAR_Variables_X*A';
%     SIGMA=1/(T)*(Error'*Error);
%     logLik=-(-0.5*T*M*(log(2*pi)+1)-0.5*T*log(det(SIGMA)));    
% 
%     end
%% Trend

A_VAR_COMP=[A(:,2:end-ExogenousRegressos);
    eye(M*(NLags-1)) zeros(M*(NLags-1),M)];

    if max(eig(A_VAR_COMP)>=1)
    logLik=100000;
    else
    Error=VAR_Variables_Y-VAR_Variables_X*A';
    SIGMA=1/(T)*(Error'*Error);
    SigmaProva = SIGMA;
%     logLik=-(-0.5*T*M*(log(2*pi))-0.5*T*log(det(SIGMA))-0.5*trace((VAR_Variables_Y-VAR_Variables_X*A')*(SIGMA)^(-1)*(VAR_Variables_Y-VAR_Variables_X*A')'));
    logLik=-(-0.5*T*M*(log(2*pi)+1)-0.5*T*log(det(SIGMA)));

    end


end