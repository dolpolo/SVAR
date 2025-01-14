%% Structural Macroeconometrics - Exam Project
% Davide Delfino (0001126595)
% Giovanni Nannini (0001128796)
% Topic 2

clc, clear

global p                                                                    % Number of lags in VAR model

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

Y=data(:,1); % Industrial Production
UF=data(:,2); % Macro Uncertainty
UM=data(:,3); % Financial Uncertainty

% Time series plot
figure(1)
plot(time, UF, 'b', 'DisplayName', 'UM');
hold on 
plot(time, UM, 'r', 'DisplayName', 'UF');
hold off
xlabel('Time');
title('Macroeconomic and Financial Uncertainty');
legend;
grid on;


%% Step 1: Rolling Window Analysis for Variance Structural Breaks

W = [UM, Y, UF];

% Set parameters
p = 4;                                                                      % Number of lags
t0 = 120;                                                                   % Initial time period for rolling wondows

% Recursive estimation
for t = t0:size(W,1)
    model_recursive = estimate(varm(3,p), W(1:t,:));                        % Estimate VAR
    residuals = infer(model_recursive, W(1:t,:));                           % Compute residuals
    Sigma_recursive(:,:,t) = cov(residuals);                                % Store covariance matrix
end

% 10-years Rolling window estimation
window_size = 10 * 12; % 10 years 
for t = window_size:size(W,1)
    model_rolling = estimate(varm(3,p), W(t-window_size+1:t,:));
    residuals = infer(model_rolling, W(t-window_size+1:t,:));
    Sigma_rolling_10(:,:,t) = cov(residuals);
end

% 15-years Rolling window estimation
window_size = 15 * 12; % 10 years 
for t = window_size:size(W,1)
    model_rolling = estimate(varm(3,p), W(t-window_size+1:t, :));
    residuals = infer(model_rolling, W(t-window_size+1:t, :));
    Sigma_rolling_15(:,:,t) = cov(residuals);
end


% Plot VAR covariance matrix elements over time
figure(2)
for i = 1:3
    for j = i:3
        SR_REC = squeeze(Sigma_recursive(i,j,120:end));
        SR_10 = squeeze(Sigma_rolling_10(i,j,120:end));
        SR_15 = squeeze(Sigma_rolling_15(i,j,180:end));
        
        subplot(3,3,(i-1)*3+j);
        plot(time(120:end), SR_REC, 'b', 'LineWidth', 1.5); hold on
        plot(time(120:end), SR_10, 'r', 'LineWidth', 1.5);
        plot(time(180:end), SR_15, 'Color', [1, 0.5, 0], 'LineWidth', 1.5);
        xline([time(284) time(569) time(716)], '--k', 'LineWidth', 1.5);    % Breakpoints: 1984:M3(t=284), 2007:M12(t=569), 2020:M3(t=716)

        if i == 1 && j == 1                                                 % Titles
            t = '$Var(U_{Mt})$';
            title(t, 'interpreter', 'latex')
        elseif i == 2 && j == 2
            t = '$Var(Y_t)$';
            title(t, 'interpreter', 'latex')
        elseif i == 3 && j == 3
            t = '$Var(U_{Ft})$';
            title(t, 'interpreter', 'latex')
        elseif i == 1 && j == 2
            t = '$Cov(U_{Mt},Y_t)$';
            title(t, 'interpreter', 'latex')
        elseif i == 1 && j == 3
            t = '$Cov(U_{Mt},U_{Ft})$';
            title(t, 'interpreter', 'latex')
        elseif i == 2 && j == 3
            t = '$Cov(Y_t,U_{Ft})$';
            title(t, 'interpreter', 'latex')
        end
    end
end


%% Step 2: Closed-form VAR Estimation

options = optimset('MaxFunEvals',200000,'TolFun',1e-1000,'MaxIter',200000,'TolX',1e-1000);


%% Step 3: Structural Parameters Estimation


%% Step 4: IRFs
