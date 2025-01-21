%% Structural Macroeconometrics - Exam Project
% Davide Delfino ()
% Giovanni Nannini (0001128796)
% Topic 2

clear
clear global
clc

addpath Functions

global NLags
global VAR_Variables_X
global VAR_Variables_Y
global T1
global T2
global T3
global T4
global Sigma_1Regime
global Sigma_2Regime
global Sigma_3Regime
global Sigma_4Regime
global StandardErrorSigma_1Regime
global StandardErrorSigma_2Regime
global StandardErrorSigma_3Regime
global StandardErrorSigma_4Regime
global CommonPI

% Variables (of interest in dataset)
% 1 Date: from 8/1960 to 7/2024
%
% 2 US Industrial Production, Total Index (source: FRED)
% 3 US Industrial Production Percent Growth Rate (log-differences)
% 4 US Industrial Production Percent Growth Rate (difference ratio)
%
% 5 UF1	   US financial uncertainty measure 1-month ahead
% 6 UF3	   US financial uncertainty measure 3-month ahead      
% 7 UF12   US financial uncertainty measure 12-month ahead
%
% 8 UM1	   US macroeconomic uncertainty measure 1-month ahead
% 9 UM3	   US macroecconomic uncertainty measure 3-month ahead
% 10 UM12   US macroeconomic uncertainty measure 12-month ahead
% // uncertainty measures taken from Sydney Ludvigson website

NLags = 4; % Number of lags of the reduced form VARs
options = optimset('MaxFunEvals',200000,'TolFun',1e-1000,'MaxIter',200000,'TolX',1e-1000);   

LimitTEST = 1.64;
LimitTEST_Apha = 0.1;

% Graphs settings
LineWidth_IRF=2;
LineWidth_IRF_BOUNDS=1;
FontSizeIRFGraph=12;
FontSizeTimeSerieGraph=14;
HorizonIRF = 60;

% Data set
DataSet = readmatrix('Topic 2 Data.xlsx');

% Break dates
TB1=284; % 1984 - M4
TB2=569; % 2007 - M12
TB3=715; % 2020 - M2

%% Data Set
DataSet = DataSet(2:end-4, [3, 5, 8]);
AllDataSet=DataSet;
M=size(DataSet,2);

Y=DataSet(:,1); % Industrial Production variable
UF=DataSet(:,2); % Financial Uncertainty variable
UM=DataSet(:,3); % Macro Uncertainty variable
DataSet = [UM Y UF];

% Creates the data for the three regimes
DataSet_1Regime=DataSet(1:TB1,:); % First regime
DataSet_2Regime=DataSet(TB1+1-NLags:TB2,:); % Second regime
DataSet_3Regime=DataSet(TB2+1-NLags:TB3,:); % Third regime
DataSet_4Regime=DataSet(TB3+1-NLags:end,:); % Last regime

% Size three regimes
T1 = size(DataSet_1Regime,1)-NLags;
T2 = size(DataSet_2Regime,1)-NLags;
T3 = size(DataSet_3Regime,1)-NLags;
T4 = size(DataSet_4Regime,1)-NLags;
TAll = size(DataSet,1)-NLags;

%% Step 2: Closed-form VAR Estimation

% Whole sample

T=TAll;

VAR_Variables_Y=DataSet(NLags+1:end,:);
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1)];
for i = 1 : NLags
Lag = DataSet(NLags+1-i:end-i,:);
VAR_Variables_X = [VAR_Variables_X Lag];
end

DuplicationMatrix = zeros(M^2,0.5*M*(M+1));
DuplicationMatrix(1,1)=1;
DuplicationMatrix(2,2)=1;
DuplicationMatrix(3,3)=1;
DuplicationMatrix(4,2)=1;
DuplicationMatrix(5,4)=1;
DuplicationMatrix(6,5)=1;
DuplicationMatrix(7,3)=1;
DuplicationMatrix(8,5)=1;
DuplicationMatrix(9,6)=1;
mDD=(DuplicationMatrix'*DuplicationMatrix)^(-1)*DuplicationMatrix';
mNN=DuplicationMatrix*mDD;

KommutationMatrix = zeros(M^2,M^2);
KommutationMatrix(1,1)=1;
KommutationMatrix(2,4)=1;
KommutationMatrix(3,7)=1;
KommutationMatrix(4,2)=1;
KommutationMatrix(5,5)=1;
KommutationMatrix(6,8)=1;
KommutationMatrix(7,3)=1;
KommutationMatrix(8,6)=1;
KommutationMatrix(9,9)=1;

NMatrix = 0.5*(eye(M^2)+KommutationMatrix);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;
% try
[Beta_LK,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options);
% catch
% Beta_LK=Beta_OLS';
% Log_LK=Likelihood_UNRESTRICTED_Error(Beta_OLS');    
% end

Log_LK_whole = Log_LK;
% Mdl = varm(M,NLags);
% EstMdl = estimate(Mdl,VAR_Variables_Y);
% Sigma_eta = EstMdl.Covariance;
% Corr_Sigma_eta = corrcov(Sigma_eta);

CommonPI=Beta_LK;
Errors=VAR_Variables_Y-VAR_Variables_X*Beta_LK';
Omega_LK=1/(T)*Errors'*Errors;
% C = cov(Errors)

SE_H = reshape(sqrt(abs(diag(HESSIAN_LK^(-1)))),M,M*NLags+1)';

% Standard errors of the reduced form parameters (autoregressive parameters)
% StandardErrors_BETA=reshape(sqrt(diag(kron(Omega_LK,(VAR_Variables_X'*VAR_Variables_X)^(-1)))),M*NLags+1,M);

% Standard errors of the reduced form parameters (covariance matrix)
StandardErrors_Omega=sqrt(diag(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)'))));
StandardErrors_Omega_M=[StandardErrors_Omega(1) StandardErrors_Omega(2) StandardErrors_Omega(3);
                        StandardErrors_Omega(2) StandardErrors_Omega(4) StandardErrors_Omega(5);
                        StandardErrors_Omega(3) StandardErrors_Omega(5) StandardErrors_Omega(6)];

% ******************************************************************************
% First Regime
% ******************************************************************************

T=T1;
DataSet=DataSet_1Regime;

VAR_Variables_Y=DataSet(NLags+1:end,:);
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1)];
for i = 1 : NLags
Lag = DataSet(NLags+1-i:end-i,:);
VAR_Variables_X = [VAR_Variables_X Lag];
end

DuplicationMatrix = zeros(M^2,0.5*M*(M+1));
DuplicationMatrix(1,1)=1;
DuplicationMatrix(2,2)=1;
DuplicationMatrix(3,3)=1;
DuplicationMatrix(4,2)=1;
DuplicationMatrix(5,4)=1;
DuplicationMatrix(6,5)=1;
DuplicationMatrix(7,3)=1;
DuplicationMatrix(8,5)=1;
DuplicationMatrix(9,6)=1;
mDD=(DuplicationMatrix'*DuplicationMatrix)^(-1)*DuplicationMatrix';
mNN=DuplicationMatrix*mDD;

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;
% try
[Beta_LK,Log_LK,exitflag,output,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options);
% catch
% Beta_LK=Beta_OLS';
% Log_LK=Likelihood_UNRESTRICTED_Error(Beta_OLS');    
% end

LK_1Regime_Sampe = Log_LK;

Errors_1Regime=VAR_Variables_Y-VAR_Variables_X*Beta_LK';
Omega_LK=1/(T)*Errors_1Regime'*Errors_1Regime;
Rho_LK = corrcov(Omega_LK);

% Standard errors of the reduced form parameters (autoregressive parameters)
StandardErrors_BETA=reshape(sqrt(diag(kron(Omega_LK,(VAR_Variables_X'*VAR_Variables_X)^(-1)))),M*NLags+1,M);
% Standard errors of the reduced form parameters (covariance matrix)
StandardErrors_Omega=sqrt(diag(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)'))));
StandardErrors_Omega_M=[StandardErrors_Omega(1) StandardErrors_Omega(2) StandardErrors_Omega(3);
                                StandardErrors_Omega(2) StandardErrors_Omega(4) StandardErrors_Omega(5);
                                StandardErrors_Omega(3) StandardErrors_Omega(5) StandardErrors_Omega(6)];



SE_Sigma_1Regime=StandardErrors_Omega;
StandardErrorSigma_1Regime=(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)')));

CompanionMatrix_1Regime=[Beta_LK(:,2:end);
                         eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            
                            
Omega_1Regime=[Omega_LK;StandardErrors_Omega_M];
Beta_1Regime=[Beta_LK'; StandardErrors_BETA];

Sigma_1Regime=Omega_LK;

% Likelihood of the VAR in the first regime
LK_1Regime=[-Log_LK];

% ******************************************************************************
% Second Regime
% ******************************************************************************

T=T2;
DataSet=DataSet_2Regime;

VAR_Variables_Y=DataSet(NLags+1:end,:);
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1)];
for i = 1 : NLags
Lag = DataSet(NLags+1-i:end-i,:);
VAR_Variables_X = [VAR_Variables_X Lag];
end

DuplicationMatrix = zeros(M^2,0.5*M*(M+1));
DuplicationMatrix(1,1)=1;
DuplicationMatrix(2,2)=1;
DuplicationMatrix(3,3)=1;
DuplicationMatrix(4,2)=1;
DuplicationMatrix(5,4)=1;
DuplicationMatrix(6,5)=1;
DuplicationMatrix(7,3)=1;
DuplicationMatrix(8,5)=1;
DuplicationMatrix(9,6)=1;
mDD=(DuplicationMatrix'*DuplicationMatrix)^(-1)*DuplicationMatrix';
mNN=DuplicationMatrix*mDD;

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;
% try
[Beta_LK,Log_LK,exitflag,output,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options);
% catch
% Beta_LK=Beta_OLS';
% Log_LK=Likelihood_UNRESTRICTED_Error(Beta_OLS');    
% end

Errors_2Regime=VAR_Variables_Y-VAR_Variables_X*Beta_LK';
Omega_LK=1/(T)*Errors_2Regime'*Errors_2Regime;
Rho_LK = corrcov(Omega_LK);

% Standard errors of the reduced form parameters (autoregressive parameters)
StandardErrors_BETA=reshape(sqrt(diag(kron(Omega_LK,(VAR_Variables_X'*VAR_Variables_X)^(-1)))),M*NLags+1,M);
% Standard errors of the reduced form parameters (Covariance matrix)
StandardErrors_Omega=sqrt(diag(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)'))));
StandardErrors_Omega_M=[StandardErrors_Omega(1) StandardErrors_Omega(2) StandardErrors_Omega(3);
                                StandardErrors_Omega(2) StandardErrors_Omega(4) StandardErrors_Omega(5);
                                StandardErrors_Omega(3) StandardErrors_Omega(5) StandardErrors_Omega(6)];

LK_2Regime_Sampe = Log_LK;
                            
SE_Sigma_2Regime=StandardErrors_Omega;
StandardErrorSigma_2Regime=(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)')));

CompanionMatrix_2Regime=[Beta_LK(:,2:end);
                         eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            
                             
Omega_2Regime=[Omega_LK;StandardErrors_Omega_M];
Beta_2Regime=[Beta_LK'; StandardErrors_BETA];

Sigma_2Regime=Omega_LK;

% Likelihood of the VAR in the second regime
LK_2Regime=[-Log_LK];
          
% ******************************************************************************
% Third Regime
% ******************************************************************************

T=T3;
DataSet=DataSet_3Regime;

VAR_Variables_Y=DataSet(NLags+1:end,:);
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1)];
for i = 1 : NLags
Lag = DataSet(NLags+1-i:end-i,:);
VAR_Variables_X = [VAR_Variables_X Lag];
end

DuplicationMatrix = zeros(M^2,0.5*M*(M+1));
DuplicationMatrix(1,1)=1;
DuplicationMatrix(2,2)=1;
DuplicationMatrix(3,3)=1;
DuplicationMatrix(4,2)=1;
DuplicationMatrix(5,4)=1;
DuplicationMatrix(6,5)=1;
DuplicationMatrix(7,3)=1;
DuplicationMatrix(8,5)=1;
DuplicationMatrix(9,6)=1;
mDD=(DuplicationMatrix'*DuplicationMatrix)^(-1)*DuplicationMatrix';
mNN=DuplicationMatrix*mDD;

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK,Log_LK,exitflag,output,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options);

LK_3Regime_Sampe = Log_LK;

Errors_3Regime=VAR_Variables_Y-VAR_Variables_X*Beta_LK';
Omega_LK=1/(T)*Errors_3Regime'*Errors_3Regime;
Rho_LK = corrcov(Omega_LK);

% Standard errors of the reduced form parameters (autoregressive parameters)
StandardErrors_BETA=reshape(sqrt(diag(kron(Omega_LK,(VAR_Variables_X'*VAR_Variables_X)^(-1)))),M*NLags+1,M);

% Standard errors of the reduced form parameters (covariance matrix)
StandardErrors_Omega=sqrt(diag(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)'))));
StandardErrors_Omega_M=[StandardErrors_Omega(1) StandardErrors_Omega(2) StandardErrors_Omega(3);
                                StandardErrors_Omega(2) StandardErrors_Omega(4) StandardErrors_Omega(5);
                                StandardErrors_Omega(3) StandardErrors_Omega(5) StandardErrors_Omega(6)];

SE_Sigma_3Regime=StandardErrors_Omega;
StandardErrorSigma_3Regime=(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)')));
                            
CompanionMatrix_3Regime=[Beta_LK(:,2:end);
                         eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            
                                       
Omega_3Regime=[Omega_LK;StandardErrors_Omega_M];
Beta_3Regime=[Beta_LK'; StandardErrors_BETA];

Sigma_3Regime=Omega_LK;

% Likelihood of the VAR in the third regime  
LK_3Regime=[-Log_LK];

% ******************************************************************************
% Fourth Regime
% ******************************************************************************

T=T4;
DataSet=DataSet_4Regime;

VAR_Variables_Y=DataSet(NLags+1:end,:);
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1)];
for i = 1 : NLags
Lag = DataSet(NLags+1-i:end-i,:);
VAR_Variables_X = [VAR_Variables_X Lag];
end

DuplicationMatrix = zeros(M^2,0.5*M*(M+1));
DuplicationMatrix(1,1)=1;
DuplicationMatrix(2,2)=1;
DuplicationMatrix(3,3)=1;
DuplicationMatrix(4,2)=1;
DuplicationMatrix(5,4)=1;
DuplicationMatrix(6,5)=1;
DuplicationMatrix(7,3)=1;
DuplicationMatrix(8,5)=1;
DuplicationMatrix(9,6)=1;
mDD=(DuplicationMatrix'*DuplicationMatrix)^(-1)*DuplicationMatrix';
mNN=DuplicationMatrix*mDD;

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK,Log_LK,exitflag,output,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options);

LK_4Regime_Sampe = Log_LK;

Errors_4Regime=VAR_Variables_Y-VAR_Variables_X*Beta_LK';
Omega_LK=1/(T)*Errors_4Regime'*Errors_4Regime;
Rho_LK = corrcov(Omega_LK);

% Standard errors of the reduced form parameters (autoregressive parameters)
StandardErrors_BETA=reshape(sqrt(diag(kron(Omega_LK,(VAR_Variables_X'*VAR_Variables_X)^(-1)))),M*NLags+1,M);

% Standard errors of the reduced form parameters (covariance matrix)
StandardErrors_Omega=sqrt(diag(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)'))));
StandardErrors_Omega_M=[StandardErrors_Omega(1) StandardErrors_Omega(2) StandardErrors_Omega(3);
                                StandardErrors_Omega(2) StandardErrors_Omega(4) StandardErrors_Omega(5);
                                StandardErrors_Omega(3) StandardErrors_Omega(5) StandardErrors_Omega(6)];

SE_Sigma_4Regime=StandardErrors_Omega;
StandardErrorSigma_4Regime=(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)')));
                            
CompanionMatrix_4Regime=[Beta_LK(:,2:end);
                         eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            
                                       
Omega_4Regime=[Omega_LK;StandardErrors_Omega_M];
Beta_4Regime=[Beta_LK'; StandardErrors_BETA];

Sigma_4Regime=Omega_LK;

% Likelihood of the VAR in the third regime  
LK_4Regime=[-Log_LK];

%% Step 3: Structural Parameters Estimation

% Upper Panel of Table 2
StructuralParam=22; 
StructuralParam_upper=22; 
InitialValue_SVAR_Initial=[
0.5;
0.5;
0;
0.5;
0.5;

0;
0.5;
0.5;
0.5;
0;
0.5;

0.5;
0.5;
0;
0.5;
0;

-0.5;
0.5;
0;
-0.5;
0
0.5]';

% InitialValue_SVAR_Initial = [0.0421063462998413;-0.0117405094668808;-0.0448104973829488;0.0368304377782917;-0.0157488924466030;0.00580239808317136;-0.0218826768384612;0.0446235881977053;0.0253441293699309;0.0590524902571742;0.0251946247347187;0.0113867188814942;-0.0446119691076574;0.0448659977279010;0.0428022339672646;0.00907094402960949];
% ML function
[StructuralParam_Estiamtion_MATRIX,Likelihood_MATRIX,exitflag,output,grad,Hessian_MATRIX] = fminunc('Likelihood_SVAR_Restricted_Upper',InitialValue_SVAR_Initial',options);

StructuralParam_Estiamtion=StructuralParam_Estiamtion_MATRIX;
LK_Estimation_upper=Likelihood_MATRIX;
Hessian_Estimation=Hessian_MATRIX;
SE_Estimation=diag(Hessian_Estimation^(-1)).^0.5;

% Overidentification LR test
LR_Test_UP = 2 * ((LK_1Regime(1)+LK_2Regime(1)+LK_3Regime(1)+LK_4Regime(1))+LK_Estimation_upper);
PVarl_UP = 1 - chi2cdf((LR_Test_UP),24-StructuralParam);

Parameters = [ [1:StructuralParam]' StructuralParam_Estiamtion SE_Estimation]; 

% Here below we define the matrices of the structural parameters with restrictions on the coefficients as described in eq. (18) of the paper. SVAR_C corresponds to the B matrix in the paper, SVAR_Q2 corresponds to
% the Q2 matrix of the paper and SVAR_Q3 corresponds to the Q3 matrix of the paper.

SVAR_C=[StructuralParam_Estiamtion(1) StructuralParam_Estiamtion(3) 0;
        StructuralParam_Estiamtion(2) StructuralParam_Estiamtion(4) 0;
        0                             0                             StructuralParam_Estiamtion(5)];

SVAR_Q2=[StructuralParam_Estiamtion(6)  0                             StructuralParam_Estiamtion(10);
         StructuralParam_Estiamtion(7)  StructuralParam_Estiamtion(9) 0;
         StructuralParam_Estiamtion(8)  0                             StructuralParam_Estiamtion(11)];

SVAR_Q3=[0                              0                              StructuralParam_Estiamtion(14);
         StructuralParam_Estiamtion(12) StructuralParam_Estiamtion(13) StructuralParam_Estiamtion(15);
         0                              0                              StructuralParam_Estiamtion(16)];

SVAR_Q4=[StructuralParam_Estiamtion(17) 0                              StructuralParam_Estiamtion(20);
         StructuralParam_Estiamtion(18) StructuralParam_Estiamtion(19) StructuralParam_Estiamtion(21);
         0                              0                              StructuralParam_Estiamtion(22)];
              
SVAR_1Regime=SVAR_C; % B
SVAR_2Regime=SVAR_C+SVAR_Q2;   % B+Q2
SVAR_3Regime=SVAR_C+SVAR_Q2+SVAR_Q3;  % B+Q2+Q3
SVAR_4Regime=SVAR_C+SVAR_Q2+SVAR_Q3+SVAR_Q4;  % B+Q2+Q3+Q4

% Flip the sign if the parameter on the main diagonal is negative

	if SVAR_1Regime(1,1)<0
    SVAR_1Regime(:,1)=-SVAR_1Regime(:,1);
    end
    if SVAR_1Regime(2,2)<0
    SVAR_1Regime(:,2)=-SVAR_1Regime(:,2); 
    end
    if SVAR_1Regime(3,3)<0
    SVAR_1Regime(:,3)=-SVAR_1Regime(:,3);
    end
    
	if SVAR_2Regime(1,1)<0
    SVAR_2Regime(:,1)=-SVAR_2Regime(:,1);
    end
    if SVAR_2Regime(2,2)<0
    SVAR_2Regime(:,2)=-SVAR_2Regime(:,2); 
    end
    if SVAR_2Regime(3,3)<0
    SVAR_2Regime(:,3)=-SVAR_2Regime(:,3);
    end
    
    if SVAR_3Regime(1,1)<0
    SVAR_3Regime(:,1)=-SVAR_3Regime(:,1);
    end
    if SVAR_3Regime(2,2)<0
    SVAR_3Regime(:,2)=-SVAR_3Regime(:,2); 
    end
    if SVAR_3Regime(3,3)<0
    SVAR_3Regime(:,3)=-SVAR_3Regime(:,3);
    end

    if SVAR_4Regime(1,1)<0
    SVAR_4Regime(:,1)=-SVAR_4Regime(:,1);
    end
    if SVAR_4Regime(2,2)<0
    SVAR_4Regime(:,2)=-SVAR_4Regime(:,2); 
    end
    if SVAR_4Regime(3,3)<0
    SVAR_4Regime(:,3)=-SVAR_4Regime(:,3);
    end
     
MATRICES=[SVAR_1Regime;
          SVAR_2Regime;
          SVAR_3Regime;
          SVAR_4Regime]
   
% Calculates the analytical derivatives organized in block matrices      
V11=2*NMatrix*kron(SVAR_C,eye(M));
V21=2*NMatrix*kron(SVAR_C,eye(M))+kron(SVAR_Q2,eye(M))+kron(eye(M),SVAR_Q2)*KommutationMatrix;
V22=kron(eye(M),SVAR_C)*KommutationMatrix+kron(SVAR_C,eye(M))+2*NMatrix*kron(SVAR_Q2,eye(M));
V31=2*NMatrix*kron(SVAR_C,eye(M))+kron(SVAR_Q2,eye(M))+kron(SVAR_Q3,eye(M))+kron(eye(M),SVAR_Q2)*KommutationMatrix+kron(eye(M),SVAR_Q3)*KommutationMatrix;
V32=kron(eye(M),SVAR_C)*KommutationMatrix+kron(SVAR_C,eye(M))+2*NMatrix*kron(SVAR_Q2,eye(M))+kron(SVAR_Q3,eye(M))+kron(eye(M),SVAR_Q3)*KommutationMatrix;
V33=kron(eye(M),SVAR_C)*KommutationMatrix+kron(eye(M),SVAR_Q2)*KommutationMatrix+2*NMatrix*kron(SVAR_Q3,eye(M))+kron(SVAR_C,eye(M))+kron(SVAR_Q2,eye(M));
V41 = 2*NMatrix*kron(SVAR_C,eye(M))+ kron(SVAR_Q2,eye(M)) + kron(SVAR_Q3,eye(M)) + kron(SVAR_Q4,eye(M))+ kron(eye(M),SVAR_Q2)*KommutationMatrix + kron(eye(M),SVAR_Q3)*KommutationMatrix+ kron(eye(M),SVAR_Q4)*KommutationMatrix;
V42 = kron(eye(M),SVAR_C)*KommutationMatrix+ kron(SVAR_C,eye(M)) + 2*NMatrix*kron(SVAR_Q2,eye(M))+ kron(SVAR_Q3,eye(M)) + kron(SVAR_Q4,eye(M))+ kron(eye(M),SVAR_Q3)*KommutationMatrix+ kron(eye(M),SVAR_Q4)*KommutationMatrix;
V43 = kron(eye(M),SVAR_C)*KommutationMatrix+ kron(eye(M),SVAR_Q2)*KommutationMatrix+ 2*NMatrix*kron(SVAR_Q3,eye(M)) + kron(SVAR_C,eye(M))+ kron(SVAR_Q2,eye(M)) + kron(SVAR_Q4,eye(M))+ kron(eye(M),SVAR_Q4)*KommutationMatrix;
V44 = kron(eye(M),SVAR_C)*KommutationMatrix+ kron(eye(M),SVAR_Q2)*KommutationMatrix + kron(eye(M),SVAR_Q3)*KommutationMatrix+ 2*NMatrix*kron(SVAR_Q4,eye(M)) + kron(SVAR_C,eye(M))+ kron(SVAR_Q2,eye(M)) + kron(SVAR_Q3,eye(M));

% Calculates the matrix for checking the rank condition (full column rank)
RankMatrix=kron(eye(4),mDD)*[V11 zeros(M^2,M^2) zeros(M^2,M^2) zeros(M^2,M^2);
                             V21 V22            zeros(M^2,M^2) zeros(M^2,M^2);
                             V31 V32            V33            zeros(M^2,M^2);
                             V41 V42            V43            V44];
 
% Selection matrix for extracting the structural parameters                         
HSelection=zeros(M*M*4,StructuralParam);
HSelection(1,1)=1;
HSelection(2,2)=1;
HSelection(4,3)=1;
HSelection(5,4)=1;
HSelection(9,5)=1;
HSelection(10,6)=1;
HSelection(11,7)=1;
HSelection(12,8)=1;
HSelection(14,9)=1;
HSelection(16,10)=1;
HSelection(18,11)=1;
HSelection(20,12)=1;
HSelection(23,13)=1;
HSelection(25,14)=1;
HSelection(26,15)=1;
HSelection(27,16)=1;
HSelection(28,17)=1;
HSelection(29,18)=1;
HSelection(32,19)=1;
HSelection(34,20)=1;
HSelection(35,21)=1;
HSelection(36,22)=1;

Jacobian= RankMatrix*HSelection;

% Report the rank of the matrix for checking the identification
rank(Jacobian)


%% Estimation of the standard errors of the parameters in B, B+Q2, B+Q2+Q3 (delta method)

StructuralEstimationCorrected=[
        MATRICES(1,1);
        MATRICES(2,1);
        MATRICES(1,2);
        MATRICES(2,2);
        MATRICES(3,3);
        MATRICES(4,1)-MATRICES(1,1);
        MATRICES(5,1)-MATRICES(2,1);
        MATRICES(6,1);
        MATRICES(5,2)-MATRICES(2,2);
        MATRICES(4,3);
        MATRICES(6,3)-MATRICES(3,3);
        MATRICES(8,1)-MATRICES(5,1);         
        MATRICES(8,2)-MATRICES(5,2);
        MATRICES(7,3)-MATRICES(4,3);
        MATRICES(8,3);
        MATRICES(9,3)-MATRICES(6,3);
        MATRICES(10,1)-MATRICES(7,1);
        MATRICES(10,3)-MATRICES(7,3);
        MATRICES(11,1)-MATRICES(8,1);
        MATRICES(11,2)-MATRICES(8,2);
        MATRICES(11,3)-MATRICES(8,3);
        MATRICES(12,3)-MATRICES(9,3);
        ];
% 
OUTPUT_Table2_StructuralEstimation=[StructuralEstimationCorrected SE_Estimation];


VAR_Est = Hessian_Estimation^(-1); 
 
% Here below we calculate the standard errors for B+Q2 and B+Q2+Q3 where we
% sum two ore three structural parameters. The indeces i,j,k refer to B,
% Q2, and Q3 respectively.

i=1; 
j=6;
index=1;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);
    
i=2;
j=7;
index=2;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=4;
j=9;
index=3;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=5;
j=11;
index=4;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=2;
j=7;
k=12;
index=5;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);
 
i=4;
j=9;
k=13;
index=6;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);
 
i=10;
j=14;
index=7;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=5;
j=11;
k=16;
index=8;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);

i=1;
j=6;
k=17;
index=9;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);

i=2;
j=7;
k=12;
s=18;
index=10;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation(s,1);
syms x y z w
f = x + y + z + w;
gradient_sigma=gradient(f, [x, y z w]);
gradient_sigma_est=subs(gradient_sigma,[x y z w],[first_par second_par third_par fourth_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k s],[i j k s])*gradient_sigma_Matrix').^0.5);

i=4;
j=9;
k=13;
s=19;
index=11;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation(s,1);
syms x y z w
f = x + y + z + w;
gradient_sigma=gradient(f, [x, y z w]);
gradient_sigma_est=subs(gradient_sigma,[x y z w],[first_par second_par third_par fourth_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k s],[i j k s])*gradient_sigma_Matrix').^0.5);

i=10;
j=14;
k=20;
index=12;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);

i=15;
j=21;
index=13;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=5;
j=11;
k=16;
s=22;
index=14;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation(s,1);
syms x y z w
f = x + y + z + w;
gradient_sigma=gradient(f, [x, y z w]);
gradient_sigma_est=subs(gradient_sigma,[x y z w],[first_par second_par third_par fourth_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k s],[i j k s])*gradient_sigma_Matrix').^0.5);

% standard error matrices

SE_ANALYTIC = [SE_Estimation; SETetaDelta];         
                         
OUTPUT_Table2_SE_Analytic = [SE_ANALYTIC(1)  SE_ANALYTIC(3)  0;
                             SE_ANALYTIC(2)  SE_ANALYTIC(4)  0;
                             0               0               SE_ANALYTIC(5);

                             SE_ANALYTIC(23) SE_ANALYTIC(3)  SE_ANALYTIC(10);
                             SE_ANALYTIC(24) SE_ANALYTIC(25) 0;
                             SE_ANALYTIC(8)  0               SE_ANALYTIC(26);   

                             SE_ANALYTIC(23) SE_ANALYTIC(3)  SE_ANALYTIC(29);
                             SE_ANALYTIC(27) SE_ANALYTIC(28) SE_ANALYTIC(15);
                             SE_ANALYTIC(8)  0               SE_ANALYTIC(30);

                             SE_ANALYTIC(31) SE_ANALYTIC(3)  SE_ANALYTIC(34);
                             SE_ANALYTIC(32) SE_ANALYTIC(33) SE_ANALYTIC(35);
                             SE_ANALYTIC(8)  0               SE_ANALYTIC(36);
                             ];

SVAR_1Regime_SE_UP = OUTPUT_Table2_SE_Analytic(1:3,:);
SVAR_2Regime_SE_UP = OUTPUT_Table2_SE_Analytic(4:6,:);
SVAR_3Regime_SE_UP = OUTPUT_Table2_SE_Analytic(7:9,:);
SVAR_4Regime_SE_UP = OUTPUT_Table2_SE_Analytic(10:12,:);

SVAR_1Regime_UP = SVAR_1Regime;
SVAR_2Regime_UP = SVAR_2Regime;
SVAR_3Regime_UP = SVAR_3Regime;
SVAR_4Regime_UP = SVAR_4Regime;


%%  Lower Panel of Table 2
StructuralParam=20; 
StructuralParam_lower=20; 
InitialValue_SVAR_Initial=0.5*ones(StructuralParam,1);

% ML function
[StructuralParam_Estiamtion_MATRIX,Likelihood_MATRIX,exitflag,output,grad,Hessian_MATRIX] = fminunc('Likelihood_SVAR_Restricted',InitialValue_SVAR_Initial',options);

StructuralParam_Estiamtion=StructuralParam_Estiamtion_MATRIX;
LK_Estimation_lower=Likelihood_MATRIX;
Hessian_Estimation=Hessian_MATRIX;

SE_Estimation=diag(Hessian_Estimation^(-1)).^0.5;

% Overidentification LR test
LR_Test_LW = 2 * ((LK_1Regime(1)+LK_2Regime(1)+LK_3Regime(1)+LK_4Regime(1))+LK_Estimation_lower);
PVar_LW = 1 - chi2cdf(LR_Test_LW,24-StructuralParam);
 
Parameters = [ [1:StructuralParam]' StructuralParam_Estiamtion' SE_Estimation]; 

% Here below we define the matrices of the structural parameters with restrictions on the coefficients as described in eq. (18) of the paper. SVAR_C corresponds to the B matrix in the paper, SVAR_Q2 corresponds to
% the Q2 matrix of the paper and SVAR_Q3 corresponds to the Q3 matrix of the paper.

SVAR_C=[StructuralParam_Estiamtion(1) 0                             0;
        StructuralParam_Estiamtion(2) StructuralParam_Estiamtion(3) 0;
        0                             0                             StructuralParam_Estiamtion(4)];

SVAR_Q2=[StructuralParam_Estiamtion(5)  0                             StructuralParam_Estiamtion(8);
         StructuralParam_Estiamtion(6)  StructuralParam_Estiamtion(7) 0;
         0                              0                             StructuralParam_Estiamtion(9)];

SVAR_Q3=[0                              0                              StructuralParam_Estiamtion(12);
         StructuralParam_Estiamtion(10) StructuralParam_Estiamtion(11) StructuralParam_Estiamtion(13);
         0                              0                              StructuralParam_Estiamtion(14)];

SVAR_Q4=[StructuralParam_Estiamtion(15) 0                              StructuralParam_Estiamtion(18);
         StructuralParam_Estiamtion(16) StructuralParam_Estiamtion(17) StructuralParam_Estiamtion(19);
         0                              0                              StructuralParam_Estiamtion(20)];
              
SVAR_1Regime=SVAR_C; % B
SVAR_2Regime=SVAR_C+SVAR_Q2;   % B+Q2
SVAR_3Regime=SVAR_C+SVAR_Q2+SVAR_Q3;  % B+Q2+Q3
SVAR_4Regime=SVAR_C+SVAR_Q2+SVAR_Q3+SVAR_Q4;  % B+Q2+Q3+Q4

% Flip the sign if the paraeter on the main diagonal is negative

	if SVAR_1Regime(1,1)<0
    SVAR_1Regime(:,1)=-SVAR_1Regime(:,1);
    end
    if SVAR_1Regime(2,2)<0
    SVAR_1Regime(:,2)=-SVAR_1Regime(:,2); 
    end
    if SVAR_1Regime(3,3)<0
    SVAR_1Regime(:,3)=-SVAR_1Regime(:,3);
    end
    
	if SVAR_2Regime(1,1)<0
    SVAR_2Regime(:,1)=-SVAR_2Regime(:,1);
    end
    if SVAR_2Regime(2,2)<0
    SVAR_2Regime(:,2)=-SVAR_2Regime(:,2); 
    end
    if SVAR_2Regime(3,3)<0
    SVAR_2Regime(:,3)=-SVAR_2Regime(:,3);
    end
    
    if SVAR_3Regime(1,1)<0
    SVAR_3Regime(:,1)=-SVAR_3Regime(:,1);
    end
    if SVAR_3Regime(2,2)<0
    SVAR_3Regime(:,2)=-SVAR_3Regime(:,2); 
    end
    if SVAR_3Regime(3,3)<0
    SVAR_3Regime(:,3)=-SVAR_3Regime(:,3);
    end

    if SVAR_4Regime(1,1)<0
    SVAR_4Regime(:,1)=-SVAR_4Regime(:,1);
    end
    if SVAR_4Regime(2,2)<0
    SVAR_4Regime(:,2)=-SVAR_4Regime(:,2); 
    end
    if SVAR_4Regime(3,3)<0
    SVAR_4Regime(:,3)=-SVAR_4Regime(:,3);
    end
     
MATRICES=[SVAR_1Regime;
          SVAR_2Regime;
          SVAR_3Regime;
          SVAR_4Regime]
   

% Calculates the analytical derivatives organized in block matrices      
V11=2*NMatrix*kron(SVAR_C,eye(M));
V21=2*NMatrix*kron(SVAR_C,eye(M))+kron(SVAR_Q2,eye(M))+kron(eye(M),SVAR_Q2)*KommutationMatrix;
V22=kron(eye(M),SVAR_C)*KommutationMatrix+kron(SVAR_C,eye(M))+2*NMatrix*kron(SVAR_Q2,eye(M));
V31=2*NMatrix*kron(SVAR_C,eye(M))+kron(SVAR_Q2,eye(M))+kron(SVAR_Q3,eye(M))+kron(eye(M),SVAR_Q2)*KommutationMatrix+kron(eye(M),SVAR_Q3)*KommutationMatrix;
V32=kron(eye(M),SVAR_C)*KommutationMatrix+kron(SVAR_C,eye(M))+2*NMatrix*kron(SVAR_Q2,eye(M))+kron(SVAR_Q3,eye(M))+kron(eye(M),SVAR_Q3)*KommutationMatrix;
V33=kron(eye(M),SVAR_C)*KommutationMatrix+kron(eye(M),SVAR_Q2)*KommutationMatrix+2*NMatrix*kron(SVAR_Q3,eye(M))+kron(SVAR_C,eye(M))+kron(SVAR_Q2,eye(M));
V41 = 2*NMatrix*kron(SVAR_C,eye(M))+ kron(SVAR_Q2,eye(M)) + kron(SVAR_Q3,eye(M)) + kron(SVAR_Q4,eye(M))+ kron(eye(M),SVAR_Q2)*KommutationMatrix + kron(eye(M),SVAR_Q3)*KommutationMatrix+ kron(eye(M),SVAR_Q4)*KommutationMatrix;
V42 = kron(eye(M),SVAR_C)*KommutationMatrix+ kron(SVAR_C,eye(M)) + 2*NMatrix*kron(SVAR_Q2,eye(M))+ kron(SVAR_Q3,eye(M)) + kron(SVAR_Q4,eye(M))+ kron(eye(M),SVAR_Q3)*KommutationMatrix+ kron(eye(M),SVAR_Q4)*KommutationMatrix;
V43 = kron(eye(M),SVAR_C)*KommutationMatrix+ kron(eye(M),SVAR_Q2)*KommutationMatrix+ 2*NMatrix*kron(SVAR_Q3,eye(M)) + kron(SVAR_C,eye(M))+ kron(SVAR_Q2,eye(M)) + kron(SVAR_Q4,eye(M))+ kron(eye(M),SVAR_Q4)*KommutationMatrix;
V44 = kron(eye(M),SVAR_C)*KommutationMatrix+ kron(eye(M),SVAR_Q2)*KommutationMatrix + kron(eye(M),SVAR_Q3)*KommutationMatrix+ 2*NMatrix*kron(SVAR_Q4,eye(M)) + kron(SVAR_C,eye(M))+ kron(SVAR_Q2,eye(M)) + kron(SVAR_Q3,eye(M));

% Calculates the matrix for checking the rank condition (full column rank)
RankMatrix=kron(eye(4),mDD)*[V11 zeros(M^2,M^2) zeros(M^2,M^2) zeros(M^2,M^2);
                             V21 V22            zeros(M^2,M^2) zeros(M^2,M^2);
                             V31 V32            V33            zeros(M^2,M^2);
                             V41 V42            V43            V44];
 
 
% Selection matrix for extracting the structural parameters                         
HSelection=zeros(M*M*4,StructuralParam);
HSelection(1,1)=1;
HSelection(2,2)=1;
HSelection(5,3)=1;
HSelection(9,4)=1;
HSelection(10,5)=1;
HSelection(11,6)=1;
HSelection(14,7)=1;
HSelection(16,8)=1;
HSelection(18,9)=1;
HSelection(20,10)=1;
HSelection(23,11)=1;
HSelection(25,12)=1;
HSelection(26,13)=1;
HSelection(27,14)=1;
HSelection(28,15)=1;
HSelection(29,16)=1;
HSelection(32,17)=1;
HSelection(34,18)=1;
HSelection(35,19)=1;
HSelection(36,20)=1;

Jacobian= RankMatrix*HSelection;

% Report the rank of the matrix for checking the identification
rank(Jacobian)

%% Estimation of the standard errors of the parameters in B, B+Q2, B+Q2+Q3 (delta method)

StructuralEstimationCorrected=[
        MATRICES(1,1);
        MATRICES(2,1);
        MATRICES(2,2);
        MATRICES(3,3);
        MATRICES(4,1)-MATRICES(1,1);
        MATRICES(5,1)-MATRICES(2,1);
        MATRICES(5,2)-MATRICES(2,2);
        MATRICES(4,3);
        MATRICES(6,3)-MATRICES(3,3);
        MATRICES(8,1)-MATRICES(5,1);         
        MATRICES(8,2)-MATRICES(5,2);
        MATRICES(7,3)-MATRICES(4,3);
        MATRICES(8,3);
        MATRICES(9,3)-MATRICES(6,3);
        MATRICES(10,1)-MATRICES(7,1);
        MATRICES(10,3)-MATRICES(7,3);
        MATRICES(11,1)-MATRICES(8,1);
        MATRICES(11,2)-MATRICES(8,2);
        MATRICES(11,3)-MATRICES(8,3);
        MATRICES(12,3)-MATRICES(9,3);];
% 
OUTPUT_Table2_StructuralEstimation=[StructuralEstimationCorrected SE_Estimation];


VAR_Est = Hessian_Estimation^(-1); 
 
% Here below we calculate the standard errors for B+Q2 and B+Q2+Q3 where we
% sum two ore three structural parameters. The indeces i,j,k refer to B,
% Q2, and Q3 respectively.

i=1; 
j=5;
index=1;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);
    
i=2;
j=6;
index=2;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=3;
j=7;
index=3;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=4;
j=9;
index=4;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=2;
j=6;
k=10;
index=5;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);
 
i=3;
j=7;
k=11;
index=6;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);
 
i=8;
j=12;
index=7;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=4;
j=9;
k=14;
index=8;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);

i=1;
j=5;
k=15;
index=9;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);

i=2;
j=6;
k=10;
s=16;
index=10;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation(s,1);
syms x y z w
f = x + y + z + w;
gradient_sigma=gradient(f, [x, y z w]);
gradient_sigma_est=subs(gradient_sigma,[x y z w],[first_par second_par third_par fourth_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k s],[i j k s])*gradient_sigma_Matrix').^0.5);

i=3;
j=7;
k=11;
s=17;
index=11;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation(s,1);
syms x y z w
f = x + y + z + w;
gradient_sigma=gradient(f, [x, y z w]);
gradient_sigma_est=subs(gradient_sigma,[x y z w],[first_par second_par third_par fourth_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k s],[i j k s])*gradient_sigma_Matrix').^0.5);

i=8;
j=12;
k=18;
index=12;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);

i=13;
j=19;
index=13;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=4;
j=9;
k=14;
s=20;
index=14;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation(s,1);
syms x y z w
f = x + y + z + w;
gradient_sigma=gradient(f, [x, y z w]);
gradient_sigma_est=subs(gradient_sigma,[x y z w],[first_par second_par third_par fourth_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k s],[i j k s])*gradient_sigma_Matrix').^0.5);

% standard error matrices

SE_ANALYTIC = [SE_Estimation; SETetaDelta];         
                         
OUTPUT_Table2_SE_Analytic = [SE_ANALYTIC(1)  0               0;
                             SE_ANALYTIC(2)  SE_ANALYTIC(3)  0;
                             0               0               SE_ANALYTIC(4);

                             SE_ANALYTIC(21) 0               SE_ANALYTIC(8);
                             SE_ANALYTIC(22) SE_ANALYTIC(23) 0;
                             0               0               SE_ANALYTIC(24);   

                             SE_ANALYTIC(21) 0               SE_ANALYTIC(27);
                             SE_ANALYTIC(25) SE_ANALYTIC(26) SE_ANALYTIC(13);
                             0               0               SE_ANALYTIC(28);

                             SE_ANALYTIC(29) 0               SE_ANALYTIC(32);
                             SE_ANALYTIC(30) SE_ANALYTIC(31) SE_ANALYTIC(33);
                             0               0               SE_ANALYTIC(34);
                             ];

SVAR_1Regime_SE = OUTPUT_Table2_SE_Analytic(1:3,:);
SVAR_2Regime_SE = OUTPUT_Table2_SE_Analytic(4:6,:);
SVAR_3Regime_SE = OUTPUT_Table2_SE_Analytic(7:9,:);
SVAR_4Regime_SE = OUTPUT_Table2_SE_Analytic(10:12,:);

SVAR_1Regime = SVAR_1Regime;
SVAR_2Regime = SVAR_2Regime;
SVAR_3Regime = SVAR_3Regime;
SVAR_4Regime = SVAR_4Regime;

%% Output Upper Panel
disp('----------------------------------------------------------------')
disp('----------------------- UPPER PANEL ----------------------------')
disp('----------------------------------------------------------------')
disp('----------------------- Coefficients ---------------------------')

disp('B=')
disp(SVAR_1Regime_UP)
disp('B+Q2=')
disp(SVAR_2Regime_UP)
disp('B+Q2+Q3=')
disp(SVAR_3Regime_UP)
disp('B+Q2+Q3+Q4=')
disp(SVAR_4Regime_UP)

disp('---------------------- Standard Errors -------------------------')

disp('B=')
disp(SVAR_1Regime_SE_UP)
disp('B+Q2=')
disp(SVAR_2Regime_SE_UP)
disp('B+Q2+Q3=')
disp(SVAR_3Regime_SE_UP)
disp('B+Q2+Q3+Q4=')
disp(SVAR_4Regime_SE_UP)


disp('---- 2 overidentification restrictions: ----')
disp('Test statistics:')
disp(LR_Test_UP)
disp('P-value:')
disp(PVarl_UP)


%% Output Lower Panel
disp('----------------------------------------------------------------')
disp('----------------------- LOWER PANEL ----------------------------')
disp('----------------------------------------------------------------')
disp('----------------------- Coefficients ---------------------------')

disp('B=')
disp(SVAR_1Regime)
disp('B+Q2=')
disp(SVAR_2Regime)
disp('B+Q2+Q3=')
disp(SVAR_3Regime)
disp('B+Q2+Q3+Q4=')
disp(SVAR_4Regime)

disp('---------------------- Standard Errors -------------------------')

disp('B=')
disp(SVAR_1Regime_SE)
disp('B+Q2=')
disp(SVAR_2Regime_SE)
disp('B+Q2+Q3=')
disp(SVAR_3Regime_SE)
disp('B+Q2+Q3+Q4=')
disp(SVAR_4Regime_SE)

disp('---- 5 overidentification restrictions ----')
disp('Test statistics:')
disp(LR_Test_LW)
disp('P-value:')
disp(PVar_LW)


disp('----------------------------------------------------------------')
disp('--------------------- MODEL COMPARISON -------------------------')
disp('----------------------------------------------------------------')
disp('----------------------- Coefficients ---------------------------')

LR_model_comp = - 2*(LK_Estimation_upper - LK_Estimation_lower);
PVar_model_comp = 1 - chi2cdf(LR_model_comp, StructuralParam_upper - StructuralParam_lower);

disp('Test statistics:')
disp(LR_model_comp)
disp('P-value:')
disp(PVar_model_comp)


%% Step 4: IRFs

BootstrapIterations = 500;
HorizonIRF = 60;  
quant = [5,95]; % quantile bootstrap to build 90% confidence intervals

global Sigma_boot_1Regime
global Sigma_boot_2Regime
global Sigma_boot_3Regime
global Sigma_boot_4Regime

for boot = 1 : BootstrapIterations

    %  **** iid bootstrap ****
    %Residuals_Boot = zeros(TAll-NLags,M);
    %Residuals_Boot(1:TB1,:) = mvnrnd(zeros(M,1),Sigma_1Regime,T1+NLags);
    %Residuals_Boot(TB1+1:TB2,:) = mvnrnd(zeros(M,1),Sigma_2Regime,T2);
    %Residuals_Boot(TB2+1:TB3,:) = mvnrnd(zeros(M,1),Sigma_3Regime,T3);
    %Residuals_Boot(TB3+1:TAll,:) = mvnrnd(zeros(M,1),Sigma_4Regime,T4-NLags);

    TBoot=datasample(1:T1,T1);
    Residuals_Boot(1:TB1-NLags,:) = Errors_1Regime(TBoot,:);
    TBoot=datasample(1:T2,T2);
    Residuals_Boot(TB1-NLags+1:TB2-NLags,:) = Errors_2Regime(TBoot,:);
    TBoot=datasample(1:T3,T3);
    Residuals_Boot(TB2-NLags+1:TB3-NLags,:) = Errors_3Regime(TBoot,:);
    TBoot=datasample(1:T4,T4);
    Residuals_Boot(TB3-NLags+1:TAll,:) = Errors_4Regime(TBoot,:);

    DataSet_Bootstrap=zeros(T1+NLags,M);
    DataSet_Bootstrap(1:NLags,:)=DataSet_1Regime(1:NLags,:);                % set the first p elements equal to the original sample values

        for t = 1+NLags : TB1
            DataSet_Bootstrap(t,:) = Beta_1Regime(1,:) + DataSet_Bootstrap(t-1,:)*Beta_1Regime(2:4,:) + ...
                                     DataSet_Bootstrap(t-2,:)*Beta_1Regime(5:7,:) + ...
                                     DataSet_Bootstrap(t-3,:)*Beta_1Regime(8:10,:) + ...
                                     DataSet_Bootstrap(t-4,:)*Beta_1Regime(11:13,:) + ...
                                     Residuals_Boot(t-NLags,:);
        end
        for t = TB1+1 : TB2
            DataSet_Bootstrap(t,:) = Beta_2Regime(1,:) + DataSet_Bootstrap(t-1,:)*Beta_2Regime(2:4,:) + ...
                                     DataSet_Bootstrap(t-2,:)*Beta_2Regime(5:7,:) + ...
                                     DataSet_Bootstrap(t-3,:)*Beta_2Regime(8:10,:) + ...
                                     DataSet_Bootstrap(t-4,:)*Beta_2Regime(11:13,:) + ...
                                     Residuals_Boot(t-NLags,:);
        end
        for t = TB2+1 : TB3
            DataSet_Bootstrap(t,:) = Beta_3Regime(1,:) + DataSet_Bootstrap(t-1,:)*Beta_3Regime(2:4,:) + ...
                                     DataSet_Bootstrap(t-2,:)*Beta_3Regime(5:7,:) + ...
                                     DataSet_Bootstrap(t-3,:)*Beta_3Regime(8:10,:) + ...
                                     DataSet_Bootstrap(t-4,:)*Beta_3Regime(11:13,:) + ...
                                     Residuals_Boot(t-NLags,:);
        end
        for t = TB3+1 : TAll
            DataSet_Bootstrap(t,:) = Beta_4Regime(1,:) + DataSet_Bootstrap(t-1,:)*Beta_4Regime(2:4,:) + ...
                                     DataSet_Bootstrap(t-2,:)*Beta_4Regime(5:7,:) + ...
                                     DataSet_Bootstrap(t-3,:)*Beta_4Regime(8:10,:) + ...
                                     DataSet_Bootstrap(t-4,:)*Beta_4Regime(11:13,:) + ...
                                     Residuals_Boot(t-NLags,:);
        end

    DataSet_Bootstrap=DataSet_Bootstrap(1+NLags:end,:);
    
    VAR = varm(M,NLags);
    [EstVAR_Boot,EstSE_Boot,logLikVAR_Boot,Residuals_Boot] = estimate(VAR,DataSet_Bootstrap(1:TB1,:));
    mP_boot_1Regime = [EstVAR_Boot.AR{1,1} EstVAR_Boot.AR{1,2} EstVAR_Boot.AR{1,3} EstVAR_Boot.AR{1,4}];
    Sigma_boot_1Regime = (Residuals_Boot'*Residuals_Boot)/T1;

    [EstVAR_Boot,EstSE_Boot,logLikVAR_Boot,Residuals_Boot] = estimate(VAR,DataSet_Bootstrap(TB1+1:TB2,:));
    mP_boot_2Regime = [EstVAR_Boot.AR{1,1} EstVAR_Boot.AR{1,2} EstVAR_Boot.AR{1,3} EstVAR_Boot.AR{1,4}];
    Sigma_boot_2Regime = (Residuals_Boot'*Residuals_Boot)/T2;

    [EstVAR_Boot,EstSE_Boot,logLikVAR_Boot,Residuals_Boot] = estimate(VAR,DataSet_Bootstrap(TB2+1:TB3,:));
    mP_boot_3Regime = [EstVAR_Boot.AR{1,1} EstVAR_Boot.AR{1,2} EstVAR_Boot.AR{1,3} EstVAR_Boot.AR{1,4}];
    Sigma_boot_3Regime = (Residuals_Boot'*Residuals_Boot)/T3;

    [EstVAR_Boot,EstSE_Boot,logLikVAR_Boot,Residuals_Boot] = estimate(VAR,DataSet_Bootstrap(TB3+1:TAll-NLags,:));
    mP_boot_4Regime = [EstVAR_Boot.AR{1,1} EstVAR_Boot.AR{1,2} EstVAR_Boot.AR{1,3} EstVAR_Boot.AR{1,4}];
    Sigma_boot_4Regime = (Residuals_Boot'*Residuals_Boot)/T4;

    
    options = optimset('MaxFunEvals',200000,'TolFun',1e-500,'MaxIter',200000,'TolX',1e-50);
    BootStructuralParam = 20;
    Initial_Structural_Boot=0.5*ones(BootStructuralParam,1);
    [StructuralParam_Estimation_Boot,Likelihood_SVAR,exitflag,output,grad,Hessian_MATRIX] = fminunc('Likelihood_SVAR_Restricted_Bootstrap',Initial_Structural_Boot',options);

    C_Boot =[StructuralParam_Estimation_Boot(1) 0                                  0;
             StructuralParam_Estimation_Boot(2) StructuralParam_Estimation_Boot(3) 0;
             0                                  0                                  StructuralParam_Estimation_Boot(4)];

    Q2_Boot=[StructuralParam_Estimation_Boot(5) 0                                  StructuralParam_Estimation_Boot(8);
             StructuralParam_Estimation_Boot(6) StructuralParam_Estimation_Boot(7) 0;
             0                                  0                                  StructuralParam_Estimation_Boot(9)];

    Q3_Boot=[0                                   0                                   StructuralParam_Estimation_Boot(12);
             StructuralParam_Estimation_Boot(10) StructuralParam_Estimation_Boot(11) StructuralParam_Estimation_Boot(13);
             0                                   0                                   StructuralParam_Estimation_Boot(14)];

    Q4_Boot=[StructuralParam_Estimation_Boot(15) 0                                   StructuralParam_Estimation_Boot(18);
             StructuralParam_Estimation_Boot(16) StructuralParam_Estimation_Boot(17) StructuralParam_Estimation_Boot(19);
             0                                   0                                   StructuralParam_Estimation_Boot(20)];

    SVAR_1Regime_Boot = C_Boot;
    SVAR_2Regime_Boot = C_Boot + Q2_Boot;
    SVAR_3Regime_Boot = C_Boot + Q2_Boot + Q3_Boot;
    SVAR_4Regime_Boot = C_Boot + Q2_Boot + Q3_Boot + Q4_Boot;

    % Sign normalization (recall that identification holds up to sign normalization)   
    if SVAR_1Regime_Boot(1,1)<0
    SVAR_1Regime_Boot(:,1)=-SVAR_1Regime_Boot(:,1);
    end
    if SVAR_1Regime_Boot(2,2)<0
    SVAR_1Regime_Boot(:,2)=-SVAR_1Regime_Boot(:,2); 
    end
    if SVAR_1Regime_Boot(3,3)<0
    SVAR_1Regime_Boot(:,3)=-SVAR_1Regime_Boot(:,3);
    end

    if SVAR_2Regime_Boot(1,1)<0
    SVAR_2Regime_Boot(:,1)=-SVAR_2Regime_Boot(:,1);
    end
    if SVAR_2Regime_Boot(2,2)<0
    SVAR_2Regime_Boot(:,2)=-SVAR_2Regime_Boot(:,2); 
    end
    if SVAR_2Regime_Boot(3,3)<0
    SVAR_2Regime_Boot(:,3)=-SVAR_2Regime_Boot(:,3);
    end

    if SVAR_3Regime_Boot(1,1)<0
    SVAR_3Regime_Boot(:,1)=-SVAR_3Regime_Boot(:,1);
    end
    if SVAR_3Regime_Boot(2,2)<0
    SVAR_3Regime_Boot(:,2)=-SVAR_3Regime_Boot(:,2); 
    end
    if SVAR_3Regime_Boot(3,3)<0
    SVAR_3Regime_Boot(:,3)=-SVAR_3Regime_Boot(:,3);
    end

    if SVAR_4Regime_Boot(1,1)<0
    SVAR_4Regime_Boot(:,1)=-SVAR_4Regime_Boot(:,1);
    end
    if SVAR_4Regime_Boot(2,2)<0
    SVAR_4Regime_Boot(:,2)=-SVAR_4Regime_Boot(:,2); 
    end
    if SVAR_4Regime_Boot(3,3)<0
    SVAR_4Regime_Boot(:,3)=-SVAR_4Regime_Boot(:,3);
    end

    J=[eye(M) zeros(M,M*(NLags-1))];
    CompanionMatrix_1Regime_Boot = [mP_boot_1Regime;
                                    eye(M*(NLags-1)) zeros(M*(NLags-1),M)];
    CompanionMatrix_2Regime_Boot = [mP_boot_2Regime;
                                    eye(M*(NLags-1)) zeros(M*(NLags-1),M)];
    CompanionMatrix_3Regime_Boot = [mP_boot_3Regime;
                                    eye(M*(NLags-1)) zeros(M*(NLags-1),M)];
    CompanionMatrix_4Regime_Boot = [mP_boot_4Regime;
                                    eye(M*(NLags-1)) zeros(M*(NLags-1),M)];

    for h = 0 : HorizonIRF
    TETA_Boot_1Regime(:,:,h+1,boot) = J*CompanionMatrix_1Regime_Boot^h*J'*SVAR_1Regime_Boot;
    TETA_Boot_2Regime(:,:,h+1,boot) = J*CompanionMatrix_2Regime_Boot^h*J'*SVAR_2Regime_Boot;
    TETA_Boot_3Regime(:,:,h+1,boot) = J*CompanionMatrix_3Regime_Boot^h*J'*SVAR_3Regime_Boot;
    TETA_Boot_4Regime(:,:,h+1,boot) = J*CompanionMatrix_4Regime_Boot^h*J'*SVAR_4Regime_Boot;
    end

end   

IRF_Inf_Boot_1Regime = prctile(TETA_Boot_1Regime,quant(1),4);
IRF_Sup_Boot_1Regime = prctile(TETA_Boot_1Regime,quant(2),4);

IRF_Inf_Boot_2Regime = prctile(TETA_Boot_2Regime,quant(1),4);
IRF_Sup_Boot_2Regime = prctile(TETA_Boot_2Regime,quant(2),4);

IRF_Inf_Boot_3Regime = prctile(TETA_Boot_3Regime,quant(1),4);
IRF_Sup_Boot_3Regime = prctile(TETA_Boot_3Regime,quant(2),4);

IRF_Inf_Boot_4Regime = prctile(TETA_Boot_4Regime,quant(1),4);
IRF_Sup_Boot_4Regime = prctile(TETA_Boot_4Regime,quant(2),4);



% ------------------------------------------------------------------------



% All 4 Volatility Regimes in one plot (No confidence bands)
% [1-blue 2-red 3-yellow 4-purple]
LineWidth_IRF = 1.5;
FontSizeIRFGraph = 14;
HorizonIRF = 60;
SubTitles = cell(1,3);

SubTitles{1,1} = '$$\varepsilon_{UM}$$';
SubTitles{1,2} = '$$\varepsilon_{Y}$$';
SubTitles{1,3} = '$$\varepsilon_{UF}$$';

YLabel = cell(3,1);
YLabel{1,1} = '$$UM$$';
YLabel{2,1} = '$$Y$$';
YLabel{3,1} = '$$UF$$';

All_IRFs = zeros(M, M, HorizonIRF+1, 4); % Dimensions: i, j, Horizon, regime

for r = 1:4
    IRF = eval(sprintf('SVAR_%dRegime', r));  
    CompanionMatrix = eval(sprintf('CompanionMatrix_%dRegime', r));
    
    J = [eye(M), zeros(M, M*(NLags-1))];

    for h = 0:HorizonIRF
        All_IRFs(:,:,h+1,r) = J * CompanionMatrix^h * J' * IRF;
    end
end


figure(5);
sgtitle('Structural IRFs for All Regimes');
index = 1;
x = 1:HorizonIRF+1;
colors = lines(4);
for i = 1:3
    for j = 1:3
        subplot(3, 3, index);
        hold on;
        for r = 1:4
            IRF = squeeze(All_IRFs(i, j, :, r));
            plot(x, IRF, 'Color', colors(r, :), 'LineWidth', LineWidth_IRF, 'DisplayName', sprintf('Regime %d', r));
        end
        plot(zeros(HorizonIRF + 1, 1), 'k', 'LineWidth', 1);

        ylabel(YLabel{i,1}, 'Interpreter', 'latex');
        title(SubTitles{1,j}, 'Interpreter', 'latex');
        set(gca, 'FontSize', FontSizeIRFGraph);
        axis tight;

        if index == 1
            legend('show', 'Location', 'best');
        end
        index = index + 1;
    end
end


% ---------------------------------------------------------------------------------------------------------------------------


% 4 Regime-specific graphs
for r = 1:4

LineWidth_IRF = 1.5;
FontSizeIRFGraph = 14;
HorizonIRF = 60;
SubTitles=cell(1,3);
SubTitles{1,1}='$$\varepsilon_{UM}$$';
SubTitles{1,2}='$$\varepsilon_{Y}$$';
SubTitles{1,3}='$$\varepsilon_{UF}$$';

YLabel=cell(3,1);
YLabel{1,1}='$$UM$$';
YLabel{2,1}='$$Y$$';
YLabel{3,1}='$$UF$$';

index = 1;

C_IRF = eval(sprintf('SVAR_%dRegime', r));                               % instantaneous impact at h=0
CompanionMatrix = eval(sprintf('CompanionMatrix_%dRegime', r));
IRF_Inf_Boot = eval(sprintf('IRF_Inf_Boot_%dRegime', r));
IRF_Sup_Boot = eval(sprintf('IRF_Sup_Boot_%dRegime', r));

J=[eye(M) zeros(M,M*(NLags-1))];                                            % selection matrix J used in IRF computation 

for h = 0 : HorizonIRF
    TETA(:,:,h+1)=J*CompanionMatrix^h*J'*C_IRF;
end

for i = 1:3
    for j = 1:3
        IRF = squeeze(TETA(i,j,:));
        IRF_Sup = squeeze(IRF_Sup_Boot(i,j,:));
        IRF_Inf = squeeze(IRF_Inf_Boot(i,j,:));

        figure(r)
        sgtitle(sprintf('Structural IRFs for Regime %d', r))
        subplot(3,3,index)
        x = 1:1:HorizonIRF+1;

        plot(x, IRF, 'LineWidth', LineWidth_IRF);
        hold all
        fill([x, fliplr(x)], [IRF_Sup', fliplr(IRF_Inf')], ...
            [0.1, 0.6, 0.9], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

        plot(zeros(HorizonIRF+1, 1), 'k', 'LineWidth', 1);
        ylabel(YLabel{i, 1}, 'Interpreter', 'latex');
        title(SubTitles{1, j}, 'Interpreter', 'latex');
        set(gca, 'FontSize', FontSizeIRFGraph);
        axis tight

        index = index + 1;
    end
end


end