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
W = [UM, Y, UF]; % Replace with your actual data


% Uncertainty indeces plot
figure
plot(time, data(:, 2), 'b', 'DisplayName', 'UM');
hold on; 
plot(time, data(:, 3), 'r', 'DisplayName', 'UF');
hold off;
xlabel('Time');
title('Time Series of UM and UF');
legend;
grid on;


%% Step 1: Descriptive analysis

varNames = {'Macroeconomic uncertainty', 'Real economic activity', 'Financial uncertainty'};

% ACF e PACF For each variable
numVars = size(W, 2);

for i = 1 : numVars
    figure
    
   % Time series graph
    subplot(3,1,1);
    plot(W(:,i));
    title(['Time Series for ', varNames{i}]);
    xlabel('Time');
    ylabel(varNames{i});
    grid on;
    
    % Autocorrelation
    subplot(3,1,2);
    autocorr(W(:,i));
    title(['Autocorrelation for ', varNames{i}]);
    
    % Partial Autocorrelation
    subplot(3,1,3);
    parcorr(W(:,i));
    title(['Partial Autocorrelation for ', varNames{i}]);
end

% Dickey-Fuller test for stationarity
numVars = size(W, 2); 
for i = 1:numVars
    [h, pValue, stat, cValue] = adftest(W(:,i));
    fprintf('Variable %d: h = %d, pValue = %.4f, stat = %.4f\n', i, h, pValue, stat);
end


%% Step 2: Rolling Window Analysis for the Identification of Volatility Regimes

% Set parameters
p = 4;                                                                      % Number of lags
t0 = 120;                                                                   % Initial time period for rolling wondows
TB1 = 284;                                                                  % First volatility break: 1984-M3
TB2 = 569;                                                                  % Second volatility break: 2007-M12
TB3 = 715;                                                                  % Third volatility break: 2020-M2


% Recursive estimation
for t = t0:size(W,1)
    model_recursive = estimate(varm(3,p), W(1:t,:));                        % Estimate VAR
    residuals = infer(model_recursive, W(1:t,:));                           % Compute residuals
    Sigma_recursive(:,:,t) = cov(residuals);                                % Store covariance matrix
end

% 10-years Rolling window estimation
window_size = 10 * 12;
for t = window_size:size(W,1)
    model_rolling = estimate(varm(3,p), W(t-window_size+1:t,:));
    residuals = infer(model_rolling, W(t-window_size+1:t,:));
    Sigma_rolling_10(:,:,t) = cov(residuals);
end

% 15-years Rolling window estimation
window_size = 15 * 12; 
for t = window_size:size(W,1)
    model_rolling = estimate(varm(3,p), W(t-window_size+1:t, :));
    residuals = infer(model_rolling, W(t-window_size+1:t, :));
    Sigma_rolling_15(:,:,t) = cov(residuals);
end


% Plot VAR covariance matrix elements over time
figure(6)
for i = 1:3
    for j = i:3
        SR_REC = squeeze(Sigma_recursive(i,j,120:end));
        SR_10 = squeeze(Sigma_rolling_10(i,j,120:end));
        SR_15 = squeeze(Sigma_rolling_15(i,j,180:end));
        
        subplot(3,3,(i-1)*3+j);
        plot(time(120:end), SR_REC, 'b', 'LineWidth', 1.5); hold on
        plot(time(120:end), SR_10, 'r', 'LineWidth', 1.5);
        plot(time(180:end), SR_15, 'Color', [1, 0.5, 0], 'LineWidth', 1.5);
        xline([time(TB1) time(TB2) time(TB3)], '--k', 'LineWidth', 1.5);    % Breakpoints: 1984:M3(t=284), 2007:M12(t=569), 2020:M2(t=715)

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


%% Step 3: Reduced-Form VAR Estimation and Diagnostic Tests 

% Define parameters
numLags = 3;
[numObs, numVars] = size(W);
alpha = 0.5;

% Whole sample estimation
Mdl = varm(numVars, numLags);
EstMdl = estimate(Mdl, W);
Sigma_eta = EstMdl.Covariance;
Corr_Sigma_eta = corrcov(Sigma_eta);

% Residuals and log-likelihood for the whole sample
Res_whole = infer(EstMdl, W);
T_whole = size(W, 1);
logLikelihood_whole = -0.5 * T_whole * (numVars * log(2 * pi) + log(det(Sigma_eta)) + trace((Res_whole' * Res_whole) / Sigma_eta));

% T-test for correlation
t_values = Corr_Sigma_eta .* sqrt(T_whole - 2) ./ sqrt(1 - Corr_Sigma_eta.^2); 
t_crit = tinv(1 - alpha/2, T_whole - 2); 
Significant_Corr_whole = abs(t_values) > t_crit;  

% Store log-likelihoods
LogLikelihoods = zeros(5, 1);
LogLikelihoods(end) = logLikelihood_whole;

% W + Time 
W_table = array2table(W, 'VariableNames', strcat("Var", string(1:size(W, 2)))); 
W_table_Date = time(2:end); 


% For subperiods, divide the data and repeat the process
subPeriod_GI = W(1:TB1, :);                                               % Mdl_GI = First subperiod   --> GI:1960:M8–1984:M3 (T = 283)
subPeriod_GM = W(TB1+1-numLags:TB2, :);                                   % Mdl_GM = Second subperiod  --> GM:1984:M4–2007:M12 (T =284)
subPeriod_GR = W(TB2+1-numLags:TB3, :);                                   % Mdl_GR = Third subperiod   --> GR+SR:2008:M1–2020:M2 (T=146)
subPeriod_C19 = W(TB3+1-numLags:end, :);                                  % Mdl_C19 = Firth subperiod  --> Covid-19 : 2020:M3–2024:M7 (T=51)

% Size three regimes
T1 = size(subPeriod_GI,1)-numLags;
T2 = size(subPeriod_GM,1)-numLags;
T3 = size(subPeriod_GR,1)-numLags;
T4 = size(subPeriod_C19,1)-numLags;
TAll = size(W,1)-numLags;

% Subperiod estimation
subPeriods = {subPeriod_GI, subPeriod_GM, subPeriod_GR, subPeriod_C19};

% Inizitialize lists
EstMdl_sub = cell(numel(subPeriods), 1);
Sigma_eta_sub = cell(numel(subPeriods), 1);
Corr_Sigma_eta_sub = cell(numel(subPeriods), 1);
Significant_Corr = cell(numel(subPeriods), 1); 

for i = 1:numel(subPeriods)
    subPeriodData = subPeriods{i};
    T_sub = size(subPeriodData, 1); 

    % Subperiods VAR estimates
    Mdl_sub = varm(numVars, numLags);
    EstMdl_sub{i} = estimate(Mdl_sub, subPeriodData);
    
    % Covariance 
    Sigma_eta_sub{i} = EstMdl_sub{i}.Covariance;
    
    % Correlation
    Corr_Sigma_eta_sub{i} = corrcov(Sigma_eta_sub{i});

    % T-test for correlations
    t_values = Corr_Sigma_eta_sub{i} .* sqrt(...
        T_sub - 2) ./ sqrt(1 - Corr_Sigma_eta_sub{i}.^2);
    t_crit = tinv(1 - alpha/2, T_sub - 2); 
    Significant_Corr{i} = abs(t_values) > t_crit; 
    
    % Residuals
    Res_sub = infer(EstMdl_sub{i}, subPeriodData);
    
    % log-likelihood
    LogLikelihoods(i) = -0.5 * T_sub * ( ...
        numVars * log(2 * pi) + log(det(Sigma_eta_sub{i})) + ...
        trace((Res_sub' * Res_sub) / Sigma_eta_sub{i}) ...
    );
end


% Display log-likelihoods
disp('Log-Likelihoods for each subperiod:');
disp(LogLikelihoods);

% Structural stability test (HO) 
LL_restricted = logLikelihood_whole;
LL_unrestricted = sum(LogLikelihoods(1:4));

% Degrees of freedom
numRestrictions = numVars^2 * (numel(subPeriods) - 1);

% LR statistic and p-value
LR_stat = -2 * (LL_restricted - LL_unrestricted);
p_value = 1 - chi2cdf(LR_stat, numRestrictions);

% Display results
fprintf('LR Test Statistic: %.4f\n', LR_stat);
fprintf('Degrees of Freedom: %d\n', numRestrictions);
fprintf('p-value: %.4f\n', p_value);

if p_value < 0.05
    disp('Reject the null hypothesis: Parameters vary across subperiods.');
else
    disp('Fail to reject the null hypothesis: Parameters are constant across subperiods.');
end

% Homoskedasticity test (H0')
Sigma_eta_residual = cov(Res_whole);
LL_whole_cov = -0.5 * T_whole * (log(det(Sigma_eta)) + trace(Sigma_eta \ Sigma_eta_residual));

LL_sub_cov = 0;
for i = 1:numel(subPeriods)
    Sigma_eta_sub = EstMdl_sub{i}.Covariance;
    T_sub = size(subPeriods{i}, 1);
    LL_sub_cov = LL_sub_cov + (-0.5 * T_sub * (log(det(Sigma_eta_sub)) + trace(Sigma_eta_sub \ Sigma_eta_residual)));
end

% LR statistic and p-value for homoskedasticity test
LR_stat_cov = - 2 * (LL_whole_cov - LL_sub_cov);
numRestrictions_cov = numVars^2 * (numel(subPeriods) - 1);
p_value_cov = 1 - chi2cdf(LR_stat_cov, numRestrictions_cov);

% Display results
fprintf('LR Test Statistic (Homoskedasticity): %.4f\n', LR_stat_cov);
fprintf('Degrees of Freedom: %d\n', numRestrictions_cov);
fprintf('p-value: %.4f\n', p_value_cov);

if p_value_cov < 0.05
    disp('Reject the null hypothesis of covariance matrix homogeneity.');
else
    disp('Fail to reject the null hypothesis of covariance matrix homogeneity.');
end

% Homoskedasticity test based on determinant ratio
det_Sigma_sub = cellfun(@(x) det(x.Covariance), EstMdl_sub);
F_stat = max(det_Sigma_sub) / min(det_Sigma_sub);

% Critical value
T_min = min(cellfun(@(x) size(x, 1), subPeriods));
T_max = max(cellfun(@(x) size(x, 1), subPeriods));
critical_value = finv(0.95, T_min - numVars, T_max - numVars);

% Display determinant ratio test results
if F_stat > critical_value
    disp('Reject the null hypothesis of homoskedasticity.');
else
    disp('Fail to reject the null hypothesis of homoskedasticity.');
end


% Doornik–Hansen Test

% Compute residuals
Res = infer(EstMdl, W);

% Number of observations and variables
[T, k] = size(Res);

% Compute sample mean and covariance of residuals
meanRes = mean(Res);
covRes = cov(Res);

% Standardize residuals
standardizedRes = (Res - meanRes) / sqrtm(covRes);

% Compute skewness and kurtosis
skewnessVec = skewness(standardizedRes);
kurtosisVec = kurtosis(standardizedRes);

% Calculate the Doornik–Hansen test statistic
skewnessTerm = T / 6 * sum(skewnessVec.^2);
kurtosisTerm = T / 24 * sum((kurtosisVec - 3).^2);
DH_stat = skewnessTerm + kurtosisTerm;

% Compute p-value from chi-squared distribution
p_value_DH = 1 - chi2cdf(DH_stat, 2 * k);

% Display results
fprintf('Doornik–Hansen Test Statistic: %.4f\n', DH_stat);
fprintf('p-value: %.4f\n', p_value_DH);

if p_value_DH < 0.05
    disp('Reject the null hypothesis: Residuals are not multivariate normal.');
else
    disp('Fail to reject the null hypothesis: Residuals are multivariate normal.');
end


% LM-Type Test for Residual Autocorrelation (AR4)

% Residuals from the VAR model
Res = infer(EstMdl_sub{1}, W);

% Maximum lag to test
maxLag = 4;

% Number of observations and variables
[T, k] = size(Res);

% Compute lagged residuals and test statistic
LM_stat = 0;
for lag = 1:maxLag
    % Lagged residuals
    laggedRes = Res(1:end-lag, :);
    currentRes = Res(lag+1:end, :);

    % Estimate the covariance between lagged and current residuals
    Gamma = (currentRes' * laggedRes) / T;

    % Compute the LM statistic
    LM_stat = LM_stat + trace(Gamma' * Gamma / cov(Res));
end

% Degrees of freedom
df = k^2 * maxLag;

% p-value
p_value_LM = 1 - chi2cdf(LM_stat, df);

% Display results
fprintf('LM Test Statistic: %.4f\n', LM_stat);
fprintf('Degrees of Freedom: %d\n', df);
fprintf('p-value: %.4f\n', p_value_LM);

if p_value_LM < 0.05
    disp('Reject the null hypothesis: Residuals are autocorrelated.');
else
    disp('Fail to reject the null hypothesis: No residual autocorrelation.');
end
