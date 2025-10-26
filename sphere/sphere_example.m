% Clear all variables from the workspace and clear the command window
clear all;
% Set the random number generator seed to 1234 for reproducible results
%rng(1234);

% Define the total number of data points (time/space samples) to generate
N = 100;
% Starting point on the sphere manifold (3D vector, must lie on the sphere)
start_point = [0; 1; 0];         
% Direction vector defining the geodesic path on the sphere (3D vector)
dir_vec = [1/sqrt(3); 0; 1/sqrt(4/3)]; 
% Create a sphere manifold object to handle sphere-specific operations (e.g., geodesic calculations)
sphere_mfd = sphere_manifold();
% Covariance matrix for input parameters (row covariance) used in data generation
cov_row = [1 0;0 1];
% Initial hyperparameters for the covariance function (log-transformed for stable optimization)
hyp_init = log([0.5,0.25]); 
% Specify the covariance function as squared exponential isotropic (handle for the function)
cov_col= @covSEiso;
% Type of data generation: 'gp' (Gaussian Process) or 'function_plus_noise'
generation_type = "gp"; % function_plus_noise; gp
% Parameters controlling the shape of functions (used if generation_type is 'function_plus_noise')
theta_params =[0.2,0.5];
% Standard deviation of Gaussian noise added to the generated output data
noise_std = 0.1;

% Number of independent trials to run (for statistical reliability of results)
num_trials = 10;
% Preallocate arrays to store prediction errors for each model across trials
gp_errors = zeros(num_trials, 1);          % Errors for iGPR model
comparison_errors = zeros(num_trials, 1);  % Errors for WGPR model
% Preallocate arrays to store computation time for each model across trials
gp_time = zeros(num_trials, 1);            % Time for iGPR model
comparison_time = zeros(num_trials, 1);    % Time for WGPR model

% Generate geodesic points on the sphere manifold and corresponding input/output data
% Inputs: sphere manifold object, number of points (N), input covariance (cov_row), 
%         covariance function (cov_col), hyperparameters (hyp_init), function params (theta_params),
%         noise level (noise_std), generation type, start point, direction vector
% Outputs: geodesic_points (points on the sphere's geodesic), x (input features), y (output data)
[geodesic_points, x, y] = sphere_generate_outputs(sphere_mfd, N, cov_row, cov_col, hyp_init, theta_params, noise_std, generation_type, start_point, dir_vec);

% Loop over each trial to evaluate model performance consistently
for trial = 1:num_trials
    % (Commented out) Optional: Regenerate data for each trial (currently uses fixed pre-generated data)
    %[geodesic_points, x, y] = sphere_generate_outputs(sphere_mfd, N, cov_row, cov_col, hyp_init, theta_params, noise_std, generation_type, start_point, dir_vec);
    % Split the dataset into training and testing subsets
    % 'random' split: randomly assign 20% of data to test set, 80% to training set;'sequential' split: randomly assign last a% of data to test set, 1-a% to training set
    % Outputs: split geodesic points (train_geo/test_geo), input features (train_x/test_x),
    %          output data (train_y/test_y), and indices of train/test samples
    
    [train_geo, test_geo, train_x, test_x, train_y, test_y, indices] = sphere_split_dataset(geodesic_points, x, y, 'random', 0.2); %sequential;random

     % Optional: Geodesic regression to estimate a prior curve 
     % p_initial = [1; 0; 0]; 
     % v_initial = [0; pi/4; 0]; 
     % lr = 0.1; 
     % iterations = 500; 
     % dim_size = 2;  
     % [train_geo, test_geo, cost] = sphere_geodesic_regression(sphere_mfd, p_initial, v_initial, train_x, train_y, test_x,lr, iterations, dim_size);

    % ---------------------- iGPR Model Prediction ----------------------
    % Start timing the computation for iGPR
    tic;
    % Predict test outputs using iGPR (Invariant Gaussian Process Regression on sphere)
    % Inputs: sphere manifold, training geodesics, training inputs, training outputs,
    %         test geodesics, test inputs
    % Outputs: iGPR_predicted_y (predicted test outputs), testL (additional output, unused here)
    [iGPR_predicted_y,testL]  = sphere_gp_prediction(sphere_mfd, train_geo, train_x, train_y, test_geo, test_x);
    % Store the computation time for this trial
    gp_time(trial) = toc; 
    % Calculate geodesic error (distance on sphere) between iGPR predictions and true test outputs
    gp_errors(trial) = sphere_geodesic_error(sphere_mfd, iGPR_predicted_y, test_y);

    % ---------------------- WGPR Model Prediction ----------------------
    % Start timing the computation for WGPR
    tic;
    % Predict test outputs using WGPR (Weighted/Alternative Gaussian Process Regression on sphere)
    WGPR_predicted_y = sphere_comparison_prediction(sphere_mfd, train_geo,train_x, train_y, test_geo, test_x);
    % Store the computation time for this trial
    comparison_time(trial) = toc;
    % Calculate geodesic error between WGPR predictions and true test outputs
    comparison_errors(trial) = sphere_geodesic_error(sphere_mfd, WGPR_predicted_y, test_y);

    % Print progress update to the command window (current trial / total trials)
    fprintf('Completed %d/%d experiments\n', trial, num_trials);
end

% Organize model names and corresponding result data for analysis
model_names = {'iGPR', 'WGPR'};                  % Names of the two models being compared
error_data = {gp_errors, comparison_errors};     % Error arrays for iGPR and WGPR
time_data = {gp_time, comparison_time};          % Time arrays for iGPR and WGPR
num_models = length(model_names);                % Number of models (2 in this case)

% Preallocate arrays to store statistical metrics (mean and standard deviation)
means = zeros(num_models, 1);      % Mean prediction error for each model
stds = zeros(num_models, 1);       % Standard deviation of prediction error
time_means = zeros(num_models, 1); % Mean computation time for each model
time_stds = zeros(num_models, 1);  % Standard deviation of computation time

% Calculate mean and standard deviation for errors and times, rounded to 4 decimal places
for i = 1:num_models
    means(i) = round(mean(error_data{i}), 4);   % Mean error
    stds(i) = round(std(error_data{i}), 4);     % Std of error
    time_means(i) = round(mean(time_data{i}), 4);% Mean time
    time_stds(i) = round(std(time_data{i}), 4);  % Std of time
end

% Create a table to summarize the statistical results
% Rows: model names; Columns: mean error, std error, mean time, std time
results_table = table(means, stds, time_means, time_stds, ...
    'RowNames', model_names, ...                 
    'VariableNames', {'Mean_Error', 'Std_Error', 'Mean_Time(s)', 'Std_Time(s)'});  
% Display the results table in the command window
disp(results_table);  


% Create a figure for visualizing prediction error distributions (size: 600x400 pixels)
figure('Position', [100, 100, 600, 400]);  
% Combine error data of both models into a single matrix for boxplot
error_data = [gp_errors, comparison_errors]; 
% Generate boxplot to compare error distributions between iGPR and WGPR
boxplot(error_data, ...
        'Labels', {'iGPR', 'WGPR'}, ...  % X-axis labels (model names)
        'Notch', 'on', ...               % Add notches (indicate 95% confidence interval for medians)
        'Whisker', 1.5, ...              % Set whisker length to 1.5×Interquartile Range (IQR)
        'Symbol', 'o', ...               % Use circles to mark outlier points
        'OutlierSize', 6);               % Set size of outlier markers to 6

hold on;  % Keep the current plot active to add additional elements

% Adjust the line width of all boxplot elements (e.g., boxes, whiskers) for better visibility
h = findobj(gca, 'Type', 'line');
for k = 1:length(h)
    set(h(k), 'LineWidth', 1.2);  
end

% Plot the mean error of each model as red diamonds (marker size 8, line width 1.5)
% Add a display name for the legend
plot(1:2, means, 'rd', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Mean');

% Add axis labels and plot title with bold font and specified size
xlabel('Prediction Models', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Mean Geodesic Error', 'FontSize', 12, 'FontWeight', 'bold');
title('Prediction Error between iGPR and WGPR (sphere)', 'FontSize', 14, 'FontWeight', 'bold');

% Add grid lines (major and minor) for easier reading of values
grid on; grid minor; 
% Add a legend in the "best" location (automatically chosen to avoid overlapping plot elements)
legend('Location', 'best', 'FontSize', 10);  

hold off;  % Release the plot (no more elements will be added)