function [predicted_y,testL] = sphere_gp_prediction(sphere_mfd, train_geo, train_t, train_y, test_geo, test_t, init_frame1)
    % Function: Perform prediction using Gaussian Process (GP) on spherical manifold
    % Core idea: Transform the nonlinear prediction problem on spherical manifold to linear regression in tangent space via manifold mapping, then map back to sphere
    % Input parameters:
    %   sphere_mfd   - Spherical manifold object (contains basic manifold operations: logarithmic map, exponential map, parallel transport, etc.)
    %   train_geo    - Training set geodesic points (3×N_train matrix, spherical 3D coordinates)
    %   train_t      - Training set time features (1×N_train row vector, used as input features for GP)
    %   train_y      - Training set response variables (3×N_train matrix, spherical target points)
    %   test_geo     - Test set geodesic points (3×N_test matrix, spherical positions to predict)
    %   test_t       - Test set time features (1×N_test row vector, used for GP prediction)
    %   init_frame   - Initial frame (3×2 matrix, orthogonal tangent vector set at a spherical point, optional parameter)
    %
    % Output parameters:
    %   predicted_y  - Prediction results (3×N_test matrix, predicted points on spherical manifold)
    
    % Get basic dimensional parameters of spherical manifold
    D = sphere_mfd.D;          % Embedding space dimension (sphere embedded in 3D space, so D=3)
    d = sphere_mfd.d;          % Manifold dimension (sphere is 2D manifold, so d=2)
    N_train = size(train_geo, 2);  % Number of training samples (obtained from second dimension)
    N_test = size(test_geo, 2);    % Number of test samples
    
    % Select initial point (default: use first geodesic point of training set as reference point)
    init_point = train_geo(:,1);  % 3×1 vector (spherical 3D coordinates


    % If initial frame not provided, automatically compute orthogonal tangent vector set at initial point
    % Orthonormal frame: 2 orthogonal tangent vectors at a spherical point, spanning 2D tangent space
    if nargin < 7 || isempty(init_frame1)
        init_frame = sphere_mfd.orthonormal_frame(init_point);  % 3×2 matrix (2 orthogonal tangent vectors)
    end
    
    % --------------------------
    % Step 1: Process training set data (convert spherical points to tangent space)
    % --------------------------
    
    % 1.1 For each training sample, compute logarithmic map from response point to geodesic point (sphere → tangent vector)
    % Logarithmic map (Log): Convert geodesic distance between two spherical points to vector in tangent space
    train_tangent_vecs = zeros(D, N_train);  % Store tangent vectors of training set (3×N_train)
    for i = 1:N_train
        geo_point = train_geo(:, i);   % Geodesic point of training sample (spherical 3D coordinates)
        y_point = train_y(:, i);       % Corresponding response point (spherical 3D coordinates)
        % Compute logarithmic map from geo_point to y_point, obtaining vector in tangent space
        train_tangent_vecs(:, i) = sphere_mfd.Log(geo_point, y_point);
    end
    
    % 1.2 Parallel transport all training set tangent vectors to tangent space of initial point (unified coordinate space)
    % Parallel transport: "Translate" tangent vector from one point to another on manifold while preserving direction
    train_tangent_at_init = zeros(D, N_train);  % Vectors in initial point's tangent space (3×N_train)
    for i = 1:N_train
        geo_point = train_geo(:, i);  % Point where original tangent vector is located
        % Parallel transport train_tangent_vecs(:,i) from geo_point to init_point
        train_tangent_at_init(:, i) = sphere_mfd.parallel_transport(...
            geo_point, init_point, train_tangent_vecs(:, i));
    end
    
    % 1.3 Convert vectors in initial point's tangent space to coordinate representation (3D tangent vector → 2D coordinates)
    % Using initial frame (orthonormal basis), project tangent vector onto basis vectors to get low-dimensional coordinates
    train_coords = zeros(d, N_train);  % Training set coordinates (2×N_train, corresponding to 2D manifold)
    for i = 1:N_train
        for j = 1:d
            % Compute inner product (manifold metric) of tangent vector with j-th basis vector as j-th coordinate
            train_coords(j, i) = sphere_mfd.metric(...
                train_tangent_at_init(:, i), init_frame(:, j));
        end
    end
    
    % --------------------------
    % Step 2: Train Gaussian Process (GP) models and  Predict test set (map from tangent space coordinates back to spherical points)
    % --------------------------
    % 2.1 Train a separate GP model for each coordinate component (total 2 dimensions), with time feature t as input
    % Use trained GP models to predict test set coordinates (input test_t,
    % output 2D coordinates) and Use trained GP models to predict test set coordinates (input test_t, output 2D coordinates)
    
    kernel = @covSEiso;
    init_func = @SE_init;
    [testL, ~] = gptp_general(train_t', train_coords', test_t', 0.01, kernel, init_func, 'All');
    test_coords = reshape(testL.mean',2,N_test);
    
    % 3.2 Convert predicted 2D coordinates to 3D tangent vectors in initial point's tangent space
    test_tangent_at_init = zeros(D, N_test);  % Predicted vectors in initial point's tangent space (3×N_test)
    for i = 1:N_test
        for j = 1:d
            % Linearly combine basis vectors with coordinate values to get tangent vector
            test_tangent_at_init(:, i) = test_tangent_at_init(:, i) + ...
                test_coords(j, i) * init_frame(:, j);
        end
    end
    
    % 3.3 Parallel transport tangent vectors from initial point to tangent spaces of corresponding test set points
    test_tangent_vecs = zeros(D, N_test);  % Vectors in test points' tangent spaces (3×N_test)
    for i = 1:N_test
        geo_point = test_geo(:, i);  % Geodesic point of test sample
        % Parallel transport test_tangent_at_init(:,i) from init_point to geo_point
        test_tangent_vecs(:, i) = sphere_mfd.parallel_transport(...
            init_point, geo_point, test_tangent_at_init(:, i));
    end
    
    % 3.4 Project tangent vectors back to spherical manifold via exponential map (tangent vector → spherical point)
    % Exponential map (Exp): Convert vector in tangent space to point on sphere
    predicted_y = zeros(D, N_test);  % Final predicted spherical points (3×N_test)
    for i = 1:N_test
        geo_point = test_geo(:, i);  % Geodesic point of test sample
        % Perform exponential map from geo_point along test_tangent_vecs(:,i) direction
        predicted_y(:, i) = sphere_mfd.Exp(geo_point, test_tangent_vecs(:, i));
    end
    
end