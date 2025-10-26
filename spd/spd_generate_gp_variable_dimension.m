function [data_matrix, time_features] = spd_generate_gp_variable_dimension(n, matD, cov_row, cov_col, hyp_init, theta_params, noise_std, generation_type)
    % Generates a sequence of symmetric positive definite (SPD) matrices over time using either Gaussian Process (GP) or function-plus-noise methods.
    % 
    % Inputs:
    %   n               - Scalar. Number of time points (and thus number of SPD matrices to generate).
    %   matD            - Scalar. Dimension of the SPD matrices (e.g., 2 for 2×2 matrices).
    %   cov_row         - Matrix. Covariance matrix for input parameters (row covariance) used in GP generation.
    %   cov_col         - Function handle. Covariance function (e.g., squared exponential) for GP generation, takes hyperparameters and inputs as arguments.
    %   hyp_init        - Vector. Initial hyperparameters for the covariance function (cov_col) in GP generation.
    %   theta_params    - Vector. Parameters controlling the shape of functions in 'function_plus_noise' generation (length 3 for 2×2 matrices).
    %   noise_std       - Scalar. Standard deviation of Gaussian noise added to the generated matrix elements.
    %   generation_type - String. Specifies the generation method: 'gp' (Gaussian Process) or 'function_plus_noise'.
    %
    % Outputs:
    %   data_matrix     - 3D array (matD×matD×n). A sequence of SPD matrices, where data_matrix(:,:,i) is the SPD matrix at the i-th time point.
    %   time_features   - Vector (1×n). Equally spaced time points from 0 to 1, corresponding to the generated matrices.
    
    % Generate time features as equally spaced points between 0 and 1
    time_features = linspace(0, 1, n);
    
    % Initialize a 3D matrix to store SPD matrices (matD×matD×n)
    data_matrix = zeros(matD, matD, n);
    
    
    % Select data generation logic based on the specified generation type
    switch lower(generation_type)
        % Gaussian Process (GP) generation: generate samples using Gaussian processes
        case 'gp'  
            % Compute the covariance matrix for time features using the specified covariance function
            C = feval(cov_col, hyp_init, time_features');
            
            % Covariance matrix for input parameters (row covariance)
            B = cov_row;
            
            % Singular Value Decomposition (SVD) of the time covariance matrix C
            [u, s, ~] = svd(C); 
            
            % Generate n samples from a multivariate normal distribution with mean 0 and covariance B
            gn = mvnrnd([0;0;0], B, n); 
            
            % Transform samples to obtain Gaussian process outputs with covariance C*B
            z_gp = u * sqrt(s) * gn;
            
            % Reshape the GP outputs into a 3×n matrix (3 independent elements for 2×2 SPD matrices)
            tangent_vectors = reshape(z_gp', 3, n);
            
            % Populate the 2×2 SPD matrix with GP-generated elements, adding noise
            data_matrix(1, 1, :) = tangent_vectors(1,:) + noise_std * randn(1,n);  % First diagonal element
            data_matrix(2, 2, :) = tangent_vectors(2,:) + noise_std * randn(1,n);  % Second diagonal element
            data_matrix(1, 2, :) = tangent_vectors(3,:) + noise_std * randn(1,n);  % Off-diagonal element (upper triangle)
            data_matrix(2, 1, :) = data_matrix(1, 2, :);  % Symmetric lower triangle (enforce symmetry)

        
        % Function-plus-noise generation: use predefined mathematical functions for matrix elements
        % Note: Only supports 2×2 matrices (3 independent elements: a, b, c for [a, b; b, c])
        case 'function_plus_noise'
            
            % Extract time points for function evaluation
            t_list = time_features;
            
            % Element (1,1): Sine-cosine combination function
            theta_a = theta_params(1);  % First parameter controls frequency
            f_a = @(t) sin(2*pi*theta_a*2*t) + 0.3*cos(2*pi*theta_a*5*t);  % Define function
            sample_a = f_a(t_list) + noise_std * randn(1,n);  % Evaluate function + add noise
            
            % Element (2,2): Exponential decay combined with sine function
            theta_c = theta_params(2);  % Second parameter controls decay rate and frequency
            f_c = @(t) exp(-t*theta_c) + 0.6*sin(2*pi*theta_c*3*t);  % Define function
            sample_c = f_c(t_list) + noise_std * randn(1,n);  % Evaluate function + add noise
            
            % Elements (1,2) and (2,1): Cosine function (symmetric elements)
            theta_b = theta_params(3);  % Third parameter controls amplitude and frequency
            f_b = @(t) 0.7*theta_b*cos(2*pi*theta_b*4*t);  % Define function
            sample_b = f_b(t_list) + noise_std * randn(1,n);  % Evaluate function + add noise
            
            % Populate the 2×2 matrix, ensuring symmetry
            data_matrix(1, 1, :) = sample_a;  % First diagonal element
            data_matrix(2, 2, :) = sample_c;  % Second diagonal element
            data_matrix(1, 2, :) = sample_b;  % Off-diagonal (upper triangle)
            data_matrix(2, 1, :) = sample_b;  % Symmetric lower triangle
            
        otherwise
            % Throw an error for unsupported generation types
            error('Unsupported generation method, please use ''gp'' or ''function_plus_noise''');
    end
end