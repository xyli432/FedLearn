function [predicted_y] = sphere_comparison_prediction(sphere_mfd, train_geo, train_t, train_y, test_geo, test_t)
    % Function: Implement extrinsic spherical Gaussian Process (GP) regression as a comparison method to predict response points on the sphere
    % Core logic: Convert spherical response points to tangent vectors via logarithmic map (extrinsic perspective processing), predict tangent vectors with GP, then map back to sphere
    % Input parameters:
    %   sphere_mfd    - Spherical manifold object (provides logarithmic map, exponential map, metric, and other manifold operations)
    %   train_geo     - Training set geodesic points (3×N_train matrix, points on sphere serving as tangent space reference points)
    %   train_t       - Training set input features (1×N_train row vector, e.g., time features)
    %   train_y       - Training set true responses (3×N_train matrix, points on sphere)
    %   test_geo      - Test set geodesic points (3×N_test matrix, points on sphere)
    %   test_t        - Test set input features (1×N_test row vector)
    % Output parameters:
    %   predicted_y   - Test set prediction results (3×N_test matrix, points on sphere)
    %   models        - Trained GP model structure array (each element corresponds to a model for one tangent vector component)
    %   projection_stats - Projection statistics (not actually assigned in function, reserved for future expansion)
    
    % -------------------------- Input parameter processing and validation --------------------------
    % Check 1: Number of geodesic points in training/test sets must match number of samples in corresponding input features
    if size(train_geo, 2) ~= length(train_t) || size(test_geo, 2) ~= length(test_t)
        error('Number of geodesic points must match number of input feature samples');
    end
    
    % Get core dimensional parameters of spherical manifold
    D = sphere_mfd.D;          % Embedding space dimension (sphere embedded in 3D space, so D=3)
    d = sphere_mfd.d;          % Manifold intrinsic dimension (sphere is 2D manifold, so d=2)
    N_train = size(train_geo, 2);  % Number of training samples
    N_test = size(test_geo, 2);    % Number of test samples
    
    % -------------------------- Step 1: Process training set data (spherical points → tangent vectors) --------------------------
    % Initialize training set tangent vector matrix (3×N_train, stores tangent vectors for each training sample)
    train_vecs = zeros(D, N_train);
    
    % Iterate through each training sample, convert response points to tangent vectors at corresponding geodesic points
    for i = 1:N_train
        geo_point = train_geo(:, i);  % Geodesic point of current training sample (tangent space reference point)
        y_point = train_y(:, i);      % True response point of current training sample (point on sphere)
        
        % Logarithmic map (Log): convert spherical point y_point to tangent vector at geo_point
        % Core extrinsic perspective operation: convert spherical nonlinear relationships to tangent space linear vectors
        tangent_vec = sphere_mfd.Log(geo_point, y_point);
        train_vecs(:, i) = tangent_vec;  % Store tangent vector
    end
    
    % -------------------------- Step 2: Train Gaussian Process (GP) models and predict --------------------------
    % 3.1 Train independent GP models for each dimension of tangent vectors (3 dimensions total, as embedded in 3D space)
    % Input: training features train_t, training tangent vectors train_vecs; Output: models (GP model array)
    % 3.2 Predict test set tangent vectors using trained GP models
    % Input: GP models, test features test_t; Output: test_vecs (predicted tangent vectors, 3×N_test)

    kernel = @covSEiso;
    init_func = @SE_init;
    [testL, ~] = gptp_general(train_t',  train_vecs', test_t', 0.01, kernel, init_func, 'All');
    test_vecs = reshape(testL.mean',3,N_test);
    
    % 3.3 Tangent vector projection and exponential map: constrain predicted tangent vectors to sphere
    predicted_y = zeros(D, N_test);  % Initialize prediction result matrix
    % Iterate through each test sample, process tangent vectors and map back to sphere
    for i = 1:N_test
        geo_point = test_geo(:, i);  % Geodesic point of current test sample (tangent space reference point)
        
        % Tangent space projection: ensure predicted tangent vectors lie strictly in tangent space of geo_point (perpendicular to geo_point)
        % Calculate inner product of tangent vector and geo_point (non-zero value indicates radial component)
        inner_prod = sphere_mfd.metric(test_vecs(:,i), geo_point);
        % Remove radial component: retain tangential component (satisfies tangent space constraint)
        tangent_vec = (test_vecs(:,i) - inner_prod*geo_point);
        % Exponential map (Exp): convert tangent vector back to point on sphere
        % Core inverse operation of extrinsic perspective: map tangent space linear vectors to spherical nonlinear points
        pred_point = sphere_mfd.Exp(geo_point,tangent_vec);
        predicted_y(:, i) = pred_point;  % Store final predicted spherical point
    end
end