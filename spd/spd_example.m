% Clear all variables from the workspace and clear the command window
clear all;
% Set the random number generator seed for reproducibility of results
%rng(1234);

% Define the number of data points to generate
N = 100;
% Specify the dimension of the SPD matrices (2x2 matrices in this case)
matD = 2;  
% Initial SPD matrix (starting point for geodesic generation)
start_mat = [2, 1; 1, 3];        
% Direction matrix used to define the geodesic path in SPD manifold
dir_mat = [4, 2; 2, 5];     

% Create an SPD manifold object of specified dimension
spd_mfd = spd(matD);                           
% Parameters for generating the output function (used in data generation)
theta_params = [0.4, 0.5, 0.3];  
% Covariance matrix for input parameters (row covariance)
%cov_row = [1 0.5 0.3;0.5 1 0.5;0.3 0.5 1];
cov_row = [1 0 0;0 1 0;0 0 1];
% Initial hyperparameters for the covariance function (log-transformed for optimization stability)
hyp_init = log([0.5,0.25]); 
% Specify the covariance function (squared exponential isotropic)
cov_col= @covSEiso;
% Type of data generation: Gaussian Process ('gp') or function plus noise
generation_type = "gp"; 
% Standard deviation of noise added to the generated data
noise_std = 0.1; 

% Number of independent trials to run for statistical robustness
num_trials = 100;
% Preallocate arrays to store errors and computation times for each trial
gp_errors = zeros(num_trials, 1);          % Errors for the first model (iGPR)
comparison_errors = zeros(num_trials, 1);  % Errors for the second model (WGPR)
comparison_errors1 = zeros(num_trials, 1); % (Unused) Errors for a potential third model
gp_time = zeros(num_trials, 1);            % Computation times for iGPR
comparison_time = zeros(num_trials, 1);    % Computation times for WGPR
comparison_time1 = zeros(num_trials, 1);   % (Unused) Times for a potential third model

% Generate geodesic points and corresponding input/output data on the SPD manifold
% Inputs: manifold, number of points, matrix dimension, covariance matrices, hyperparameters,
%         function parameters, start/direction matrices, noise level, generation type
% Outputs: geodesic points, input variables (x), output variables (y)
[geodesic_points,x, y] = spd_generate_outputs(spd_mfd, N, matD, cov_row, cov_col, hyp_init, theta_params, start_mat, dir_mat,noise_std, generation_type);

% Run multiple trials to evaluate model performance
for trial = 1:num_trials
    % Split the dataset into training and testing sets
    % 'random' split with 20% of data used for testing; 'sequential' split with a of data used for testing (the last 1-a of samples as test set)
    % Returns training/testing geodesic points, input variables (t), output variables (y), and split indices
    %[geodesic_points,x, y] = spd_generate_outputs(spd_mfd, N, matD, cov_row, cov_col, hyp_init, theta_params, start_mat, dir_mat,noise_std, generation_type);
    [train_geo, test_geo, train_t, test_t, train_y, test_y, indices] = spd_split_dataset(geodesic_points, x, y, 'random', 0.2); %sequential

    % Geodesic regression to estimate prior curve
    %[~, ~,train_t, test_t, train_y, test_y, indices] = spd_split_dataset(geodesic_points, x, y, 'random',0.2);
    %options = struct();
    %options.iterations = 200; 
    %options.lr = 0.5;         
    %options.verbose = true;    
    %[train_geo,test_geo,train_costs] = spd_geodesic_regression(train_t, train_y, x, indices.train_idx, indices.test_idx, matD, options);

    % ---------------------- iGPR Prediction ----------------------
    % Measure computation time for iGPR prediction
    tic;
    % Predict test outputs using iGPR (intrinsic Gaussian Process Regression on SPD manifold)
    % Inputs: manifold, training geodesics, training inputs, training outputs, 
    %         test geodesics, test inputs
    % Outputs: predicted test outputs, additional output (unused)
    [predicted_y,~] = spd_gp_prediction(spd_mfd, train_geo, train_t, train_y, test_geo, test_t);
    % Store the computation time for this trial
    gp_time(trial) = toc; 
    % Calculate and store the geodesic error between predictions and true test outputs
    gp_errors(trial) = spd_geodesic_error(spd_mfd, predicted_y, test_y);
    
    % ---------------------- WGPR Prediction ----------------------
    % Measure computation time for WGPR prediction
    tic;
    % Predict test outputs using WGPR (Wrapped Gaussian Process Regression variant on SPD manifold)
    [comparison_pred,~] = spd_comparison_prediction(spd_mfd, train_geo, train_t, train_y, test_geo, test_t);
    % Store the computation time for this trial
    comparison_time(trial) = toc;
    % Calculate and store the geodesic error for WGPR
    comparison_errors(trial) = spd_geodesic_error(spd_mfd,comparison_pred, test_y);
    % Print progress update
    fprintf('Completed %d/%d experiments\n', trial, num_trials);
end

% Organize results for visualization and analysis
model_names = {'iGPR', 'WGPR'};          % Names of the models being compared
error_data = {gp_errors, comparison_errors};  % Error results for each model
time_data = {gp_time, comparison_time};        % Time results for each model
num_models = length(model_names);               % Number of models

% Preallocate arrays to store mean and standard deviation of errors and times
means = zeros(num_models, 1);         % Mean error for each model
stds = zeros(num_models, 1);          % Standard deviation of error
time_means = zeros(num_models, 1);    % Mean computation time
time_stds = zeros(num_models, 1);     % Standard deviation of computation time

% Calculate mean and standard deviation for errors and times
for i = 1:num_models
    means(i) = round(mean(error_data{i}), 4);   % Round to 4 decimal places for readability
    stds(i) = round(std(error_data{i}), 4);     
    time_means(i) = round(mean(time_data{i}), 4);
    time_stds(i) = round(std(time_data{i}), 4);
end

% Create a table to summarize results with row names (model names) and column labels
results_table = table(means, stds, time_means, time_stds, ...
    'RowNames', {'iGPR', 'WGPR'}, ...                 % Row names match model names
    'VariableNames', {'Mean_Error', 'Std_Error', 'Mean_Time(s)', 'Std_Time(s)'});  % Column labels
% Display the results table in the command window
disp(results_table);  

% % Create a figure to visualize prediction errors with specified size
% figure('Position', [100, 100, 600, 400]);  
% % Combine error data into a matrix for boxplot
% error_data = [gp_errors, comparison_errors]; 
% % Generate boxplot to compare error distributions between models
% boxplot(error_data, ...
%         'Labels', {'iGPR', 'WGPR'}, ...  % Label x-axis with model names
%         'Notch', 'on', ...               % Add notches to boxes (indicate 95% CI for medians)
%         'Whisker', 1.5, ...              % Set whisker length to 1.5*IQR
%         'Symbol', 'o', ...               % Use circles for outliers
%         'OutlierSize', 6);               % Set size of outlier markers
% hold on;  % Keep plot active to add additional elements
% 
% % Adjust line width of boxplot elements for better visibility
% h = findobj(gca, 'Type', 'line');
% for k = 1:length(h)
%     set(h(k), 'LineWidth', 1.2);  
% end
% 
% % Plot mean error values as red diamonds for each model
% plot(1:2, means, 'rd', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Mean');
% 
% % Add labels and title with appropriate formatting
% xlabel('Prediction Models', 'FontSize', 12, 'FontWeight', 'bold');
% ylabel('Mean Geodesic Error', 'FontSize', 12, 'FontWeight', 'bold');
% title('Prediction Error between iGPR and WGPR (SPD)', 'FontSize', 14, 'FontWeight', 'bold');
% % Add grid for better readability
% grid on; grid minor; 
% % Add legend in the best possible location
% legend('Location', 'best', 'FontSize', 10);  
% hold off;  % Release plot

