function [train_geo, test_geo, train_t, test_t, train_y, test_y, indices] = sphere_split_dataset(geodesic_points, t, y, split_method, param)
    % Function: Split dataset on spherical manifold into training and testing sets
    % Application scenario: Separating data for model training and evaluation in spherical manifold learning or regression tasks
    % Input parameters:
    %   geodesic_points - Set of points on the geodesic (3×N matrix, 3 represents spherical 3D coordinates, N is total number of samples)
    %   t               - Time features (1×N row vector, time or index feature corresponding to each sample)
    %   y               - Response variable (3×N matrix, spherical target point coordinates corresponding to each sample)
    %   split_method    - Splitting method (string):
    %                     'random': Random split (randomly select test set from all samples)
    %                     'sequential': Sequential split (first half as training set, second half as test set in sample order)
    %   param           - Splitting parameter (corresponding to method):
    %                     For 'random': Proportion of total samples in test set (e.g., 0.2 means 20% as test set)
    %                     For 'sequential': Number of samples in training set (e.g., 50 means first 50 as training set)
    %
    % Output parameters:
    %   train_geo, test_geo - Geodesic points of training and testing sets (3×N_train and 3×N_test matrices)
    %   train_t, test_t     - Time features of training and testing sets (1×N_train and 1×N_test row vectors)
    %   train_y, test_y     - Response variables of training and testing sets (3×N_train and 3×N_test matrices)
    %   indices             - Structure containing split indices:
    %                         indices.train_idx: Original indices of training set samples (1×N_train)
    %                         indices.test_idx: Original indices of test set samples (1×N_test)
    
    % -------------------------- Input data validation --------------------------
    % Get total number of samples (read from second dimension of response variable y, N is total samples)
    N = size(y, 2);
    % Verify that the number of samples in geodesic points matches y (avoid input data length mismatch)
    N1 = size(geodesic_points, 2);
    if size(t, 2) ~= N1 || size(y, 2) ~= N1
        error('All input data must have the same number of samples');  % Throw error if inconsistent
    end
    
    % Verify that spherical data dimensions conform to 3D spherical point format (spherical points should be 3D coordinates)
    if size(geodesic_points, 1) ~= 3 || size(geodesic_points, 2) ~= N
        error('Geodesic points must be in 3×N spherical point format');  % First dimension must be 3 (x,y,z coordinates)
    end
    if size(y, 1) ~= 3 || size(y, 2) ~= N
        error('Response variables must be in 3×N spherical point format');  % Response variables must also be 3D coordinates
    end
    
    % Initialize index structure (for storing original indices of training and testing sets)
    indices = struct();
    
    % -------------------------- Method 1: Random split --------------------------
    if strcmp(split_method, 'random')  % Check if in random split mode
        % Validate parameter (proportion must be between 0 and 1, excluding 0 and 1)
        if param <= 0 || param >= 1
            error('Parameter for random split must be a proportion between (0,1)');
        end
        
        % Calculate number of test samples (total samples × proportion, rounded to nearest integer)
        test_size = round(N * param);
        % Boundary handling: Ensure at least 1 test sample and no more than total samples-1 (guarantee non-empty training set)
        if test_size == 0
            test_size = 1;  % Avoid empty test set
        elseif test_size == N
            test_size = N - 1;  % Avoid empty training set
        end
        
        % Randomly select test set indices: randomly pick test_size unique indices from 1~N
        indices.test_idx = randperm(N, test_size);
        % Training set indices: remaining samples after removing test set indices
        indices.train_idx = setdiff(1:N, indices.test_idx);
        % indices.train_idx = sort(indices.train_idx);  % (Optional) Sort training indices to maintain original order
    
    % -------------------------- Method 2: Sequential split --------------------------
    elseif strcmp(split_method, 'sequential')  % Check if in sequential split mode
        % Validate parameter (number of training samples must be positive integer less than total samples)
        if param <= 0 || param >= N
            error('Parameter for sequential split must be an integer between (0,N)');
        end
        
        % Sequential split: first param samples as training set, remaining as test set
        indices.train_idx = 1:param;          % Training set indices: 1 to param
        indices.test_idx = (param+1):N;       % Test set indices: param+1 to N
        % Re-validate to ensure non-empty test set (avoid param=N causing empty test set)
        if isempty(indices.test_idx)
            error('Number of training samples cannot equal total number of samples');
        end
    
    % -------------------------- Invalid method handling --------------------------
    else
        error('Split method must be either ''random'' or ''sequential''');  % Only two split methods supported
    end
    
    % -------------------------- Execute data splitting --------------------------
    % Split geodesic points: extract training and test sets based on indices
    train_geo = geodesic_points(:, indices.train_idx);  % Training set geodesic points (3×N_train)
    test_geo = geodesic_points(:, indices.test_idx);    % Test set geodesic points (3×N_test)
    
    % Split time features: extract time features for corresponding indices
    train_t = t(indices.train_idx);  % Training set time features (1×N_train)
    test_t = t(indices.test_idx);    % Test set time features (1×N_test)

    % Split response variables: extract target points for corresponding indices
    train_y = y(:, indices.train_idx);  % Training set response variables (3×N_train)
    test_y = y(:, indices.test_idx);    % Test set response variables (3×N_test)
end