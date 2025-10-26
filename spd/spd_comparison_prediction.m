function [predicted_y, testL] = spd_comparison_prediction(spd_mfd, train_geo, train_t, train_y, test_geo, test_t)
    % Function: Comparison method for SPD manifold prediction (simplified version)
    % Core logic: Perform GP regression directly on upper triangular elements (including diagonal) of tangent space vectors, skipping "parallel transport" step
    % Key difference from iGPR: Does not uniformly transport tangent vectors to initial point's tangent space, directly models independent elements of tangent vectors at each geodesic point
    % Input parameters:
    %   spd_mfd       - SPD manifold object (provides log/exponential maps, matrix-vector conversion, and other basic operations)
    %   train_geo     - Training set geodesic points (dimension: matD×matD×N_train)
    %   train_t       - Training set time features (dimension: 1×N_train)
    %   train_y       - Training set response variables (points on SPD manifold, dimension: matD×matD×N_train)
    %   test_geo      - Test set geodesic points (dimension: matD×matD×N_test)
    %   test_t        - Test set time features (dimension: 1×N_test)
    % Output parameters:
    %   predicted_y   - Test set prediction results (points on SPD manifold, dimension: matD×matD×N_test)
    %   models        - Trained GP model structure array (each upper triangular element corresponds to one model)
    
    % -------------------------- Basic parameter calculation --------------------------
    matD = size(train_geo, 1);          % Row and column dimension of SPD matrix (e.g., 2 for 2×2 matrices)
    N_train = size(train_geo, 3);       % Number of training samples
    N_test = size(test_geo, 3);         % Number of test samples
    d = matD * (matD + 1) / 2;          % Number of upper triangular elements in tangent vector (including diagonal, i.e., number of independent elements)
    
    % -------------------------- Step 1: Training set tangent vector processing --------------------------
    % 1.1 Project training set response variables y to tangent space of corresponding geodesic points (logarithmic map: manifold point → tangent vector)
    % Storage format: matD×matD×N_train (each training sample corresponds to one tangent vector matrix)
    train_tangent_vecs = zeros(matD, matD, N_train);
    for i = 1:N_train
        geo_mat = train_geo(:, :, i);  % i-th training geodesic point (matrix format)
        y_mat = train_y(:, :, i);      % i-th training response point (matrix format)
        
        % Logarithmic map needs vector format for calculation, convert and restore:
        geo_vec = spd_mfd.mat_to_vec(geo_mat);  % Matrix → vector (matches manifold method input)
        y_vec = spd_mfd.mat_to_vec(y_mat);      % Matrix → vector
        tangent_vec = spd_mfd.Log(geo_vec, y_vec);  % Logarithmic map: manifold point → tangent vector (vector format)
        train_tangent_vecs(:, :, i) = spd_mfd.vec_to_mat(tangent_vec);  % Vector → matrix (facilitates upper triangular extraction)
    end
    
    % 1.2 Extract upper triangular elements (including diagonal) of tangent vector matrices to form low-dimensional coordinate matrix
    % Core: Tangent vectors are symmetric matrices, upper triangular elements contain all information to avoid redundancy
    train_vecs = zeros(d, N_train);  % Coordinate matrix storing upper triangular elements (d×N_train)
    for i = 1:N_train
        triu_elems = triu(train_tangent_vecs(:, :, i));  % Extract upper triangular part of i-th tangent vector
        train_vecs(:, i) = triu_elems(triu_elems ~= 0)';  % Vectorize non-zero elements (i.e., upper triangular elements) and transpose to column vector
    end
    
    % -------------------------- Step 2: Train Gaussian Process (GP) models and  Test set prediction --------------------------
    % Train a separate GP model for each upper triangular element (total d independent elements) for multi-output modeling
    %fprintf('Starting training of comparison method GP models...\n');
    % 3.1 Predict upper triangular element coordinates for test set using trained GP models
    %fprintf('Starting test set prediction for comparison method...\n');
   
    kernel = @covSEiso;
    init_func = @SE_init;
    [testL, ~] = gptp_general(train_t', train_vecs', test_t', 0.01, kernel, init_func, 'All');
    test_vecs = reshape(testL.mean',d, N_test);

    % 3.2 Reconstruct predicted upper triangular elements into symmetric tangent vector matrices
    test_tangent_vecs = zeros(matD, matD, N_test);  % Store test set tangent vector matrices (matD×matD×N_test)
    for i = 1:N_test
        sym_mat = zeros(matD, matD);  % Initialize empty matrix
        triu_idx = triu(true(matD));  % Get upper triangular indices (logical matrix with true for upper triangle)
        sym_mat(triu_idx) = test_vecs(:, i);  % Fill upper triangular elements (using predicted coordinates)
        % Symmetrization: lower triangular elements = transpose of upper triangular elements (ensures tangent vector is symmetric matrix)
        sym_mat = sym_mat + sym_mat' - diag(diag(sym_mat));  % Subtract diagonal to avoid double counting
        test_tangent_vecs(:, :, i) = sym_mat;  % Store reconstructed tangent vector matrix
    end
    
    % 3.3 Exponential map: project tangent vectors back to SPD manifold (obtain final prediction points)
    predicted_y = zeros(matD, matD, N_test);  % Store final prediction results (matD×matD×N_test)
    for i = 1:N_test
        geo_mat = test_geo(:, :, i);  % i-th test geodesic point (matrix format)
        geo_vec = spd_mfd.mat_to_vec(geo_mat);  % Matrix → vector (matches manifold method)
        % Tangent vector matrix → vector for exponential map
        tangent_vec = spd_mfd.mat_to_vec(test_tangent_vecs(:, :, i));
        % Exponential map: tangent vector → SPD manifold point (vector format)
        pred_vec = spd_mfd.Exp(geo_vec, tangent_vec);
        predicted_y(:, :, i) = spd_mfd.vec_to_mat(pred_vec);  % Vector → matrix (final output format)

    end
    
    %fprintf('Comparison method prediction completed: test set sample count = %d\n', N_test);
end