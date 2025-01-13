%% Structural Macroeconometrics - Exam Project
% Davide Delfino ()
% Giovanni Nannini (0001128796)
% Topic 2

clc, clear

global NLags

dataTable = readtable('Topic 2 Data.xlsx');
time = datetime(dataTable{2:end-4, 1}, 'InputFormat', 'yyyy-MM-dd');
data = dataTable{2:end-4, [3, 5, 8]};


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

Y=data(:,1); % Industrial Production variable
UM=data(:,2); % Macro Uncertainty variable
UF=data(:,3); % Financial Uncertainty variable

figure(1);

% Plot delle variabili UM e UF
plot(time, data(:, 2), 'b', 'DisplayName', 'UM'); % Prima colonna di data per UM
hold on; 
plot(time, data(:, 3), 'r', 'DisplayName', 'UF'); % Seconda colonna di data per UF
hold off;

% Personalizzazione del grafico
xlabel('Time'); % Etichetta asse x
ylabel('Value'); % Etichetta asse y
title('Time Series of UM and UF'); % Titolo del grafico
legend; % Mostra la legenda
grid on; % Attiva la griglia

%% Step 1: Identification of VAR Variance Structural Breaks
 
NLags = 4; % Number of lags of the reduced form VARs
options = optimset('MaxFunEvals',200000,'TolFun',1e-1000,'MaxIter',200000,'TolX',1e-1000);




%% Step 2: Closed-form VAR Estimation



%% Step 3: Structural Parameters Estimation


%% Step 4: IRFs