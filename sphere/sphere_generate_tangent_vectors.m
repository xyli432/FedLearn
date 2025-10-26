function [tangent_vectors, time_features] = sphere_generate_tangent_vectors(n, cov_row, cov_col, hyp_init, theta_params, noise_std, generation_type)
    % Generates tangent vectors on a sphere manifold over a sequence of time points using either Gaussian Process (GP) or function-plus-noise methods.
    % 
    % Inputs:
    %   n               - Scalar. Number of time points (and thus number of tangent vectors to generate).
    %   cov_row         - Matrix. Covariance matrix for input parameters (row covariance) used in GP generation.
    %   cov_col         - Function handle. Covariance function (e.g., squared exponential) for GP generation, taking hyperparameters and inputs as arguments.
    %   hyp_init        - Vector. Initial hyperparameters for the covariance function (cov_col) in GP generation.
    %   theta_params    - Vector. Parameters controlling the shape of analytical functions in 'function_plus_noise' generation (length 2).
    %   noise_std       - Scalar. Standard deviation of Gaussian noise added to the generated tangent vector components.
    %   generation_type - String. Specifies the generation method: 'gp' (Gaussian Process) or 'function_plus_noise'.
    %
    % Outputs:
    %   tangent_vectors - 2×n Matrix. Tangent vectors on the sphere manifold, where each column corresponds to a time point.
    %   time_features   - 1×n Vector. Equally spaced time points from 0 to 1, corresponding to the generated tangent vectors.
    
    % Generate time features as equally spaced points between 0 and 1
    time_features = linspace(0, 1, n);

    % Initialize a 2×n matrix to store tangent vectors (2 dimensions for sphere manifold)
    tangent_vectors = zeros(2, n); 
    
    % Select tangent vector generation logic based on the specified method
    switch lower(generation_type)
        % Method 1: Generate tangent vectors using Gaussian Process (GP)
        case 'gp'
            % Generate multivariate GP samples using the specified covariance function, row covariance, time features, and hyperparameters
            a = mv_gptp_sample(cov_col, cov_row, time_features', hyp_init);

            % Reshape the GP output to a 2×n matrix of tangent vectors
            tangent_vectors = reshape(a', 2, n);

            % Add Gaussian noise to each component of the tangent vectors
            tangent_vectors(1,:) = tangent_vectors(1,:) + noise_std * randn(1, n);  % First component with noise
            tangent_vectors(2,:) = tangent_vectors(2,:) + noise_std * randn(1, n);  % Second component with noise

        % Method 2: Generate tangent vectors using analytical functions plus noise
        case 'function_plus_noise'
            % Extract parameters for the analytical functions from theta_params
            theta1 = theta_params(1);  % Parameter controlling the first function's shape
            theta2 = theta_params(2);  % Parameter controlling the second function's shape
            
            % Define the first analytical function (sine-cosine combination) and generate the first component of tangent vectors
            f1 = @(t) sin(2*pi*theta1*2*t) + 0.3*cos(2*pi*theta1*5*t);
            tangent_vectors(1, :) = f1(time_features) + noise_std * randn(1, n);  % Add noise to the first component
            
            % Define the second analytical function (exponential decay plus sine) and generate the second component of tangent vectors
            f2 = @(t) exp(-t*theta2) + 0.6*sin(2*pi*theta2*3*t);
            tangent_vectors(2, :) = f2(time_features) + noise_std * randn(1, n);  % Add noise to the second component
            
        % Handle unsupported generation methods with an error
        otherwise
            error('Unsupported generation method, please use ''gp'' or ''function_plus_noise''');
    end
end