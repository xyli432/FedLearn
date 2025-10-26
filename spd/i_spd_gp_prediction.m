function [predicted_y] = i_spd_gp_prediction(spd_mfd,train_geo, train_t, train_y, test_geo, test_t)
    % Function to predict SPD matrix outputs using a Gaussian Process (GP) on the SPD manifold
    % Inputs:
    %   spd_mfd      - Object containing SPD manifold operations (e.g., Log, Exp, mat_to_vec, etc.)
    %   train_geo    - Training geometric features (SPD matrices), size [matD x matD x N_train]
    %   train_t      - Training time/input variables, size [1 x N_train] or similar
    %   train_y      - Training target SPD matrices, size [matD x matD x N_train]
    %   test_geo     - Test geometric features (SPD matrices), size [matD x matD x N_test]
    %   test_t       - Test time/input variables, size [1 x N_test] or similar
    % Output:
    %   predicted_y  - Predicted SPD matrices for test data, size [matD x matD x N_test]

    matD = size(train_geo, 1);          % Dimension of SPD matrix (e.g., 2 for 2×2 matrix)
    D = matD * matD;                    % Dimension after vectorization of SPD matrix (ambient space dimension)
    d = matD * (matD + 1) / 2;          % Dimension of SPD manifold tangent space (number of independent elements in symmetric matrix)
    N_train = size(train_geo, 3);       % Number of training samples
    N_test = size(test_geo, 3);         % Number of test samples
   
    train_coords = zeros(d, N_train);   % Storage for training coordinates in tangent space
    geo_frame_list = zeros(D, d, N_train); % Storage for orthonormal frames at each training geometric feature
    for i = 1:N_train
        geo_vec = spd_mfd.mat_to_vec(train_geo(:, :, i)); % Convert training geometric feature to vector form
        y_vec = spd_mfd.mat_to_vec(train_y(:, :, i));     % Convert training target to vector form
        [geo_frame_raw, ~] = spd_mfd.orthonormal_frame(geo_vec);  % Get orthonormal frame at the geometric feature
        geo_frame = squeeze(geo_frame_raw);  % Remove singleton dimensions from the frame
        geo_frame_list(:,:,i) = geo_frame;   % Store the frame for current training sample
        % Compute coordinates of target in the tangent space of the geometric feature, using the frame
        train_coords(:,i) = reshape(spd_mfd.coef_process(geo_vec, spd_mfd.Log(geo_vec, y_vec), reshape(geo_frame, D, 1, d)), d, 1); 
    end
    
    kernel = @covSEiso;                 % Choose squared exponential kernel for GP
    init_func = @SE_init;               % Initialization function for SE kernel parameters
    % Perform GP regression on the tangent space coordinates using time/input variables
    [testL, ~] = gptp_general(train_t',  train_coords', test_t', 0.1, kernel, init_func, 'All');
    test_coords = reshape(testL.mean',d, N_test);  % Reshape predicted means to test coordinates (d x N_test)

    test_tangent_at_init = zeros(D, N_test);  % Storage for tangent vectors at test geometric features
    for i = 1:N_test
        geo_vec1 = spd_mfd.mat_to_vec(test_geo(:, :, i));  % Convert test geometric feature to vector form
        [geo_frame_raw1, ~] = spd_mfd.orthonormal_frame(geo_vec1);  % Get orthonormal frame at test geometric feature
        init_frame = squeeze(geo_frame_raw1);  % Remove singleton dimensions from the test frame
        % Reconstruct tangent vector from predicted coordinates using the test frame
        for j = 1:d
            test_tangent_at_init(:, i) = test_tangent_at_init(:, i) + ...
                test_coords(j, i) * init_frame(:, j);
        end
    end

    test_tangent_vecs = reshape(test_tangent_at_init, D, N_test);  % Reshape tangent vectors (D x N_test)
    predicted_y = zeros(matD, matD, N_test);  % Storage for final predicted SPD matrices
    for i = 1:N_test
        geo_vec = spd_mfd.mat_to_vec(test_geo(:, :, i));  % Convert test geometric feature to vector form
        pred_vec = spd_mfd.Exp(geo_vec, test_tangent_vecs(:, i));  % Map tangent vector to SPD manifold using exponential map
        predicted_y(:, :, i) = spd_mfd.vec_to_mat(pred_vec);  % Convert predicted vector back to matrix form
    end
end