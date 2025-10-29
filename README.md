# iGPR
iGPR is a MATLAB code implementing the intrinsic Gaussian Process Regression model
## two baselines

- iGPR: [intrinsic Gaussian Process Regression on Riemannian Manifolds](https://arxiv.org/abs/2411.18989)
- WGPR: [Wrapped Gaussian Process Regression on Riemannian Manifolds](https://openaccess.thecvf.com/content_cvpr_2018/html/Mallasto_Wrapped_Gaussian_Process_CVPR_2018_paper.html)

## some example commands
It includes code examples for two common manifolds, with detailed instructions on specific usage provided in the files spd-example.m and sphere-example.m.

- Before conducting the experiment, first add the paths: Run
- [startup.m](https://github.com/xyli432/iGPR/blob/main/gpml-matlab-master/startup.m)  and
- [add_path.m](https://github.com/xyli432/iGPR/blob/main/gptp_multi_output-master/add_path.m).

```
### Sphere manifold
% Clear all variables from the workspace and clear the command window
clear all;
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
hyp_init = log([1,0.1]); 
% Specify the covariance function as squared exponential isotropic (handle for the function)
cov_col= @covSEiso;
% Type of data generation: 'gp' (Gaussian Process) or 'function_plus_noise'
generation_type = "gp"; % function_plus_noise; gp
% Parameters controlling the shape of functions (used if generation_type is 'function_plus_noise')
theta_params =[0.2,0.5];
% Standard deviation of Gaussian noise added to the generated output data
noise_std = 0.1;
% Generate geodesic points on the sphere manifold and corresponding input/output data
% Inputs: sphere manifold object, number of points (N), input covariance (cov_row), 
%         covariance function (cov_col), hyperparameters (hyp_init), function params (theta_params),
%         noise level (noise_std), generation type, start point, direction vector
% Outputs: geodesic_points (points on the sphere's geodesic), x (input features), y (output data)
[geodesic_points, x, y] = sphere_generate_outputs(sphere_mfd, N, cov_row, cov_col, hyp_init, theta_params, noise_std, generation_type, start_point, dir_vec);
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
% Predict test outputs using iGPR (Invariant Gaussian Process Regression on sphere)
% Inputs: sphere manifold, training geodesics, training inputs, training outputs,
%         test geodesics, test inputs
% Outputs: iGPR_predicted_y (predicted test outputs), testL (additional output, unused here)
[iGPR_predicted_y,testL]  = sphere_gp_prediction(sphere_mfd, train_geo, train_x, train_y, test_geo, test_x);
% Calculate geodesic error (distance on sphere) between iGPR predictions and true test outputs
disp(sphere_geodesic_error(sphere_mfd, iGPR_predicted_y, test_y));

% ---------------------- WGPR Model Prediction ----------------------
% Predict test outputs using WGPR (Weighted/Alternative Gaussian Process Regression on sphere)
WGPR_predicted_y = sphere_comparison_prediction(sphere_mfd, train_geo,train_x, train_y, test_geo, test_x);
% Calculate geodesic error between WGPR predictions and true test outputs
disp(sphere_geodesic_error(sphere_mfd, WGPR_predicted_y, test_y));

### Spd manifold
% Clear all variables from the workspace and clear the command window
clear all;
% Set the random number generator seed for reproducibility of results
rng(1234);

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
cov_row = [1 0 0;0 1 0;0 0 1];
% Initial hyperparameters for the covariance function (log-transformed for optimization stability)
hyp_init = log([0.5,0.25]); 
% Specify the covariance function (squared exponential isotropic)
cov_col= @covSEiso;
% Type of data generation: Gaussian Process ('gp') or function plus noise
generation_type = "gp"; 
% Standard deviation of noise added to the generated data
noise_std = 0.1;

[train_geo, test_geo, train_t, test_t, train_y, test_y, indices] = spd_split_dataset(geodesic_points, x, y, 'random', 0.2);

% Geodesic regression to estimate prior curve
%[~, ~,train_t, test_t, train_y, test_y, indices] = spd_split_dataset(geodesic_points, x, y, 'random',0.2);
%options = struct();
%options.iterations = 200; 
%options.lr = 0.5;         
%options.verbose = true;
%[train_geo,test_geo,train_costs] = spd_geodesic_regression(train_t, train_y, x, indices.train_idx, indices.test_idx, matD, options);


% Predict test outputs using iGPR (intrinsic Gaussian Process Regression on SPD manifold)
% Inputs: manifold, training geodesics, training inputs, training outputs, 
%  test geodesics, test inputs
% Outputs: predicted test outputs, additional output (unused)
[predicted_y,~] = spd_gp_prediction(spd_mfd, train_geo, train_t, train_y, test_geo, test_t);
% Calculate and store the geodesic error between predictions and true test outputs
diap(spd_geodesic_error(spd_mfd, predicted_y, test_y));

% Predict test outputs using WGPR (Wrapped Gaussian Process Regression variant on SPD manifold)
%[comparison_pred,~] = spd_comparison_prediction(spd_mfd, train_geo, train_t, train_y, test_geo, test_t);
% Calculate and store the geodesic error for WGPR
disp(spd_geodesic_error(spd_mfd,comparison_pred, test_y));
```

## full examples 

- a full sphere example is in [sphere-example.m ](https://github.com/xyli432/iGPR/blob/main/sphere/sphere_example.m)
- a full spd example is in [spd-example.m ](https://github.com/xyli432/iGPR/blob/main/spd/spd_example.m)
