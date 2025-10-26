function [predicted_y,testL] = spd_gp_prediction(spd_mfd, train_geo, train_t, train_y, test_geo, test_t, init_frame)
    % SPD manifold prediction based on Gaussian Process (corrected version)
    % Core process: Transform prediction problem on SPD manifold into Gaussian Process regression in tangent space, then project back to manifold
    % Step breakdown: Training set tangent vector processing → Multi-component GP regression → Test set prediction → Tangent vector transport → Exponential map projection back to manifold
    % Input parameters:
    %   spd_mfd       - SPD manifold object (provides manifold operation methods like log/exponential maps, parallel transport, etc.)
    %   train_geo     - Training set geodesic points (matD×matD×N_train, reference points on SPD manifold)
    %   train_t       - Training set features (1×N_train, e.g., time series features)
    %   train_y       - Training set target SPD matrices (matD×matD×N_train)
    %   test_geo      - Test set geodesic points (matD×matD×N_test)
    %   test_t        - Test set features (1×N_test)
    %   init_frame    - Orthonormal frame at initial point (optional, D×d, D=matD², d=tangent space dimension)
    % Output parameters:
    %   predicted_y   - Test set prediction results (matD×matD×N_test, SPD matrices)
    %   models        - Trained Gaussian Process model array (total d models, one for each tangent space component)
    %   init_frame    - Orthonormal frame at initial point (automatically calculated or reused from input)
    %   test_sigma    - Test set prediction standard deviations (d×N_test, prediction uncertainty for each component)
    
    % -------------------------- Basic parameter calculation --------------------------
    matD = size(train_geo, 1);          % Dimension of SPD matrix (e.g., 2 for 2×2 matrix)
    D = matD * matD;                    % Dimension after vectorization of SPD matrix (ambient space dimension)
    d = matD * (matD + 1) / 2;          % Dimension of SPD manifold tangent space (number of independent elements in symmetric matrix)
    N_train = size(train_geo, 3);       % Number of training samples
    N_test = size(test_geo, 3);         % Number of test samples
    
    % Select first geodesic point of training set as "initial point" (reference for unified tangent vector transport)
    init_point = train_geo(:, :, 1);
    init_point_vec = spd_mfd.mat_to_vec(init_point);  % Convert to vector format (matches manifold operations)
    
    % If initial orthonormal frame not provided, automatically calculate orthonormal frame at initial point (standard basis of tangent space)
    if nargin < 7 || isempty(init_frame)
        [init_frame_raw, ~] = spd_mfd.orthonormal_frame(init_point_vec);  % Get frame (D×1×d)
        init_frame = squeeze(init_frame_raw);  % Reduce dimensions to D×d
    end
    
    % -------------------------- Step 1: Training set tangent vector processing --------------------------
    % 1.1 Project training targets y to tangent spaces of corresponding geodesic points (logarithmic map: manifold point → tangent vector)
    train_tangent_vecs = zeros(D, N_train);  % Store tangent vectors for each training sample (D×N_train)
    for i = 1:N_train
        geo_vec = spd_mfd.mat_to_vec(train_geo(:, :, i));  % i-th geodesic point (vector format)
        y_vec = spd_mfd.mat_to_vec(train_y(:, :, i));      % i-th target point (vector format)
        % Logarithmic map: calculate tangent vector from geodesic point to target point (in geodesic point's tangent space)
        train_tangent_vecs(:, i) = spd_mfd.Log(geo_vec, y_vec);
    end
    
    % 1.2 Parallel transport all tangent vectors to "initial point's" tangent space (unify tangent space for GP modeling)
    train_tangent_at_init = zeros(D, 1, N_train);  % Store transported tangent vectors (D×1×N_train)
    for i = 1:N_train
        geo_vec = spd_mfd.mat_to_vec(train_geo(:, :, i));  % i-th geodesic point (vector format)
        % Parallel transport: move train_tangent_vecs(:,i) from geo_vec's tangent space to initial point's tangent space
        train_tangent_at_init(:, :, i) = spd_mfd.parallel_transport(...
            geo_vec, init_point_vec, train_tangent_vecs(:, i));
    end
    
    % 1.3 Convert tangent vectors to "coordinates in initial frame" (high-dimensional tangent vector → low-dimensional coordinates for regression)
    % Replace nested loop implementation from original comments with batch processing via coef_process
    train_coords = reshape(...
        spd_mfd.coef_process(init_point_vec, train_tangent_at_init, reshape(init_frame, D, 1, d)), ...
        d, N_train);  % Get d×N_train coordinate matrix (each component corresponds to one GP model)
    
    % -------------------------- Step 2: Train Gaussian Process (GP) models and Test set prediction--------------------------
    % Train separate GP models for each coordinate component in tangent space (total d components) (simplified implementation of multi-output GP)
    %fprintf('Starting GP model training...\n');
    % 3.1 Predict test set coordinates using trained GP models (separate prediction for each component)
    %fprintf('Starting test set prediction...\n');

    kernel = @covSEiso;
    init_func = @SE_init;
    [testL, ~] = gptp_general(train_t', train_coords', test_t', 0.01, kernel, init_func, 'All');
    test_coords = reshape(testL.mean',d,N_test);
    
    % 3.2 Convert predicted coordinates to "tangent vectors in initial point's tangent space" (coordinates → tangent vectors)
    test_tangent_at_init = zeros(D, N_test);  % Store predicted tangent vectors in initial point's tangent space (D×N_test)
    for i = 1:N_test
        for j = 1:d
            % Coordinate weighted sum: each coordinate component × corresponding frame basis vector, accumulate to get complete tangent vector
            test_tangent_at_init(:, i) = test_tangent_at_init(:, i) + ...
                test_coords(j, i) * init_frame(:, j);
        end
    end
    test_tangent_at_init = reshape(test_tangent_at_init, D, N_test);  % Ensure correct dimensions
    
    % 3.3 Parallel transport tangent vectors from initial point to "test set geodesic points' tangent spaces"
    test_tangent_vecs = zeros(D, N_test);  % Store predicted tangent vectors in test points' tangent spaces (D×N_test)
    for i = 1:N_test
        geo_vec = spd_mfd.mat_to_vec(test_geo(:, :, i));  % i-th test geodesic point (vector format)
        % Parallel transport: move test_tangent_at_init(:,i) from initial point's tangent space to test point's tangent space
        test_tangent_vecs(:, i) = spd_mfd.parallel_transport(...
            init_point_vec, geo_vec, test_tangent_at_init(:, i));
    end
    
    % 3.4 Exponential map: project test point tangent vectors back to SPD manifold (get final prediction points)
    predicted_y = zeros(matD, matD, N_test);  % Store final prediction results (matD×matD×N_test)
    for i = 1:N_test
        geo_vec = spd_mfd.mat_to_vec(test_geo(:, :, i));  % i-th test geodesic point (vector format)
        % Exponential map: get predicted point on manifold from test point tangent vector (vector format)
        pred_vec = spd_mfd.Exp(geo_vec, test_tangent_vecs(:, i));
        predicted_y(:, :, i) = spd_mfd.vec_to_mat(pred_vec);  % Convert to matrix format
    end
    
    %fprintf('Prediction completed: Number of test samples = %d\n', N_test);
end