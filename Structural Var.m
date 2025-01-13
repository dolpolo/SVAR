%% Structural Macroeconometrics - Exam Project
% Davide Delfino ()
% Giovanni Nannini (0001128796)
% Topic 2

clc, clear

global NLags


data = readmatrix("Topic 2 Data.xlsx");
data = data(2:end-4, [3 5 8]);

% Variables (of interest in dataset)
% 1 Date: from 8/1960 to 7/2023
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

Y=DataSet(:,1); % Industrial Production variable
UM1=DataSet(:,2); % Macro Uncertainty variable
UMF=DataSet(:,3); % Financial Uncertainty variable


%% Step 1: Identification of VAR Variance Structural Breaks
 
NLags = 4; % Number of lags of the reduced form VARs
options = optimset('MaxFunEvals',200000,'TolFun',1e-1000,'MaxIter',200000,'TolX',1e-1000);




%% Step 2: Closed-form VAR Estimation



%% Step 3: Structural Parameters Estimation


%% Step 4: IRFs