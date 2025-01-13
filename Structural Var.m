%% Structural Macroeconometrics - Exam Project
% Davide Delfino (0001126595)
% Giovanni Nannini (0001128796)
% Topic 2

clc, clear

global p

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
UF=data(:,2); % Macro Uncertainty variable
UM=data(:,3); % Financial Uncertainty variable

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


%% Rolling Window

% Step 1: EStimate the VAR recursivelly
% Xt =Π(t)Wt +𝜂t, Σ𝜂(t):=E(𝜂t𝜂′t), t = 1, …,T,

% Define your data matrix
W = [UM, Y, UF]; % Replace with your actual data

% Set parameters
p = 4; % Number of lags
t0 = 120; % Initial window size (adjust as needed)

% Recursive estimation
for t = t0:size(W, 1)
    model_recursive = estimate(varm(3, p), W(1:t, :)); % Estimate VAR
    residuals = infer(model_recursive, W(1:t, :)); % Compute residuals
    Sigma_recursive(:,:,t) = cov(residuals); % Store covariance matrix
end

% Rolling window estimation
window_size = 10 * 12; % 10 years 
for t = window_size:size(W, 1)
    model_rolling = estimate(varm(3, p), W(t-window_size+1:t, :));
    residuals = infer(model_rolling, W(t-window_size+1:t, :));
    Sigma_rolling(:,:,t) = cov(residuals);
end

% Rolling window estimation
window_size = 15 * 12; % 10 years 
for t = window_size:size(W, 1)
    model_rolling = estimate(varm(3, p), W(t-window_size+1:t, :));
    residuals = infer(model_rolling, W(t-window_size+1:t, :));
    Sigma_rolling_15(:,:,t) = cov(residuals);
end

% Step 2: Plot

for i = 1 :3
    for j = 1 : 3
        SR_10 = squeeze(Sigma_rolling(i,j,:));
        figure(2);
        plot(time,SR_10)
    end
end

% Loop per creare i grafici combinati
figure;
for i = 1:3
    for j = i:3 % Solo parte triangolare superiore
        subplot(3, 3, (i - 1) * 3 + j); % Posizione del grafico nella matrice 3x3
        
        % Estrai la serie temporale per la covarianza/varianza specifica
        SR_10 = squeeze(Sigma_rolling(i, j, :));
        
        % Disegna il grafico
        plot(time, SR_10, 'LineWidth', 1.5);
        hold on;
        title(['Element (', num2str(i), ',', num2str(j), ')']);
        
        % Aggiungi etichette solo per l'ultima riga/colonna per chiarezza
        if i == 3
            xlabel('Time');
        end
        if j == 1
            ylabel('Value');
        end
    end
end

% Miglioramento generale del layout
sgtitle('Matrice di Covarianze: Parte Triangolare Inferiore');

%% Step 1: Identification of VAR Variance Structural Breaks
 

options = optimset('MaxFunEvals',200000,'TolFun',1e-1000,'MaxIter',200000,'TolX',1e-1000);




%% Step 2: Closed-form VAR Estimation



%% Step 3: Structural Parameters Estimation


%% Step 4: IRFs
