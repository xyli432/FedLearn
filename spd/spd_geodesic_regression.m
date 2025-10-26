function [train_geo, test_geo, train_costs] = spd_geodesic_regression(x_train, y_train, x_test, Indices_train, Indices_test, matD, options)
    % Function: Implement geodesic regression model on SPD manifold (fitting data by optimizing geodesic parameters)
    % Core idea: Fit data on SPD manifold with a geodesic, optimize geodesic's starting point and direction using gradient descent
    % Input parameters:
    %   x_train       - Training set features (1×N row vector, N is number of training samples)
    %   y_train       - Training set targets (matD×matD×N collection of SPD matrices)
    %   x_test        - Test set features (1×M row vector, M is number of test samples)
    %   Indices_train - Training set sample indices (for extracting training portion from predictions)
    %   Indices_test  - Test set sample indices (for extracting test portion from predictions)
    %   matD          - Dimension of SPD matrix (e.g., 2 for 2×2 matrix)
    %   options       - Optimization options structure containing:
    %                   - iterations: number of iterations (default 100)
    %                   - lr: learning rate (default 0.1)
    %                   - verbose: whether to print iteration information (default true)
    % Output parameters:
    %   train_geo    - Predicted geodesic points for training set (matD×matD×N_train)
    %   test_geo     - Predicted geodesic points for test set (matD×matD×N_test)
    %   train_costs  - Loss curve during training (1×iterations vector)
    
    % -------------------------- 1. Parameter initialization and input validation --------------------------
    % Validate input feature dimensions (must be row vectors)
    if size(x_train, 1) ~= 1 || size(x_test, 1) ~= 1
        error('x_train and x_test must be 1×N and 1×M row vectors');
    end
    % Validate training target dimensions (must be matD×matD×N SPD matrices)
    if size(y_train, 1) ~= matD || size(y_train, 2) ~= matD
        error('y_train must be matD×matD×N SPD matrices');
    end
    
    % Set default values for optimization options (if not provided)
    if nargin < 6
        options = struct();  % Initialize empty structure
    end
    if ~isfield(options, 'iterations'), options.iterations = 100; end  % Default 100 iterations
    if ~isfield(options, 'lr'), options.lr = 0.1; end                  % Default learning rate 0.1
    if ~isfield(options, 'verbose'), options.verbose = true; end        % Default to print iteration info
    
    % Calculate sample count and dimension parameters
    N = length(x_train);  % Number of training samples
    M = length(x_test);   % Number of test samples
    D = matD^2;           % Dimension after vectorization of SPD matrix (ambient space dimension)
    
    % Initialize SPD manifold object (provides basic manifold operation methods)
    spd_mfd = spd(matD);
    
    % -------------------------- 2. Model parameter initialization --------------------------
    % Initial starting point P: select first training sample as initial value (point on manifold)
    P = y_train(:, :, 1);
    P_vec = spd_mfd.mat_to_vec(P);  % Convert to vector format (facilitates manifold operations)
    
    % Initial direction V: initialize based on direction between first two samples (vector in tangent space)
    if N >= 2
        % If at least 2 samples, initialize with direction between first two samples
        y1_vec = spd_mfd.mat_to_vec(y_train(:, :, 1));  % First sample vector
        y2_vec = spd_mfd.mat_to_vec(y_train(:, :, 2));  % Second sample vector
        % Calculate tangent vector between two points (logarithmic map) and divide by feature difference to get direction
        V_init = spd_mfd.Log(y1_vec, y2_vec) / (x_train(2) - x_train(1));
    else
        % If only 1 sample, randomly initialize direction and ensure symmetry (tangent vectors must be symmetric)
        V_init = spd_mfd.mat_to_vec(0.1 * (eye(matD) + randn(matD)));
        V_init = (V_init + V_init') / 2;  % Symmetrization
    end
    V = V_init / norm(V_init);  % Normalize direction vector
    
    % Initialize loss curve storage
    train_costs = zeros(1, options.iterations);  % Training loss
    
    % -------------------------- 3. Gradient descent optimization of model parameters --------------------------
    if options.verbose
        fprintf('Starting SPD geodesic regression optimization (%d iterations)...\n', options.iterations);
    end
    
    for iter = 1:options.iterations
        % Calculate gradients for parameters P and V (call internal helper functions)
        grad_P_mat = grad_p(spd_mfd, P, V, x_train, y_train, matD, D);  % Gradient for starting point P (matrix format)
        grad_P_vec = spd_mfd.mat_to_vec(grad_P_mat);                    % Convert to vector format
        grad_V = grad_v(spd_mfd, P, V, x_train, y_train, matD, D);      % Gradient for direction V (vector format)
        
        % Update starting point P: move on manifold using exponential map (ensures remains SPD matrix after update)
        P_new_vec = spd_mfd.Exp(P_vec, -options.lr * grad_P_vec);  % Move in negative gradient direction
        P_new = spd_mfd.vec_to_mat(P_new_vec);                     % Convert back to matrix format
        
        % Update direction V: first update by gradient descent, then parallel transport to new starting point's tangent space
        V_new = V - options.lr * grad_V;  % Gradient descent update
        % Parallel transport: move V_new from original starting point P's tangent space to new starting point P_new's tangent space
        V_new = spd_mfd.parallel_transport(P_vec, P_new_vec, V_new);
        
        % Update parameter values
        P = P_new;
        P_vec = P_new_vec;
        V = V_new;
        
        % Calculate training loss (average geodesic distance)
        y_pred_train = predict(spd_mfd, P_vec, V, x_train, matD);  % Training set predictions
        train_costs(iter) = compute_cost(spd_mfd, y_pred_train, y_train);  % Calculate loss
        
        % Calculate test set predictions (for subsequent extraction of train_geo and test_geo)
        y_pred = predict(spd_mfd, P_vec, V, x_test, matD);
        
        % Print iteration progress (every 10 iterations)
        if options.verbose && mod(iter, 10) == 0
            fprintf('Iteration %d | Training loss: %.6f \n', iter, train_costs(iter));
        end

        % Extract training and test set predictions based on indices
        train_geo = y_pred(:, :, Indices_train);
        test_geo = y_pred(:, :, Indices_test);
    end
end


% -------------------------- Internal helper functions --------------------------

% Function: Calculate average geodesic distance between predictions and true values (loss function)
function cost = compute_cost(spd_mfd, y_pred, y_true)
    N = size(y_true, 3);  % Number of samples
    cost = 0;             % Initialize loss
    
    % Accumulate geodesic distance for each sample
    for i = 1:N
        % Call manifold object's distance function (geodesic distance)
        cost = cost + spd_mfd.dist(y_pred(:, :, i), y_true(:, :, i));
    end
    cost = cost / N;  % Average loss
end

% Function: Calculate gradient for starting point parameter P (based on gradient descent on manifold)
function grad_P = grad_p(spd_mfd, P, V, x, y, matD, D)
    N = length(x);              % Number of training samples
    grad_P_vec = zeros(D, 1);   % Initialize gradient vector (vector format)
    P_vec = spd_mfd.mat_to_vec(P);  % Vector format of starting point P
    
    for i = 1:N
        % 1. Generate predicted point for i-th sample (point on geodesic)
        t = x(i);                  % Feature value of i-th sample
        v_scaled = t * V;          % Scale direction vector by feature value
        mapped_value_vec = spd_mfd.Exp(P_vec, v_scaled);  % Exponential map: from tangent vector to manifold point
        
        % 2. Calculate error between predicted and true points (logarithmic map)
        y_i_vec = spd_mfd.mat_to_vec(y(:, :, i));  % Vector format of true point
        error_vec = spd_mfd.Log(mapped_value_vec, y_i_vec);  % Logarithmic map: error vector (in predicted point's tangent space)
        
        % 3. Parallel transport error vector to starting point P's tangent space (unify tangent space for gradient calculation)
        error_transported = spd_mfd.parallel_transport(mapped_value_vec, P_vec, error_vec);
        
        % 4. Accumulate gradients (negative direction of average error)
        grad_P_vec = grad_P_vec + error_transported;
    end
    grad_P_vec = -grad_P_vec / N;  % Average gradient, with negative sign (gradient descent direction)
    grad_P = spd_mfd.vec_to_mat(grad_P_vec);  % Convert to matrix format
end

% Function: Calculate gradient for direction parameter V (based on gradient descent on manifold)
function grad_V = grad_v(spd_mfd, P, V, x, y, matD, D)
    N = length(x);              % Number of training samples
    grad_V = zeros(D, 1);       % Initialize gradient vector
    P_vec = spd_mfd.mat_to_vec(P);  % Vector format of starting point P
    
    for i = 1:N
        % 1. Generate predicted point for i-th sample (point on geodesic)
        t = x(i);                  % Feature value of i-th sample
        v_scaled = t * V;          % Scale direction vector by feature value
        mapped_value_vec = spd_mfd.Exp(P_vec, v_scaled);  % Exponential map: from tangent vector to manifold point
        
        % 2. Calculate error between predicted and true points (logarithmic map)
        y_i_vec = spd_mfd.mat_to_vec(y(:, :, i));  % Vector format of true point
        error_vec = spd_mfd.Log(mapped_value_vec, y_i_vec);  % Logarithmic map: error vector (in predicted point's tangent space)
        
        % 3. Parallel transport error vector to starting point P's tangent space
        error_transported = spd_mfd.parallel_transport(mapped_value_vec, P_vec, error_vec);
        
        % 4. Accumulate gradients (error multiplied by feature value, reflecting feature's influence on direction)
        grad_V = grad_V + error_transported * x(i);
    end
    grad_V = -grad_V / N;  % Average gradient, with negative sign (gradient descent direction)
end

% Function: Predict points on geodesic based on current model parameters (P and V)
function y_pred = predict(spd_mfd, P_vec, V, x, matD)
    N = length(x);  % Number of samples
    % Initialize prediction results (matD×matD×N)
    y_pred = zeros(matD, matD, N);
    
    for i = 1:N
        t = x(i);                  % Feature value of i-th sample
        v_scaled = t * V;          % Scale direction vector by feature value
        y_pred_vec = spd_mfd.Exp(P_vec, v_scaled);  % Exponential map: from tangent vector to manifold point
        y_pred(:, :, i) = spd_mfd.vec_to_mat(y_pred_vec);  % Convert to matrix format
    end
end