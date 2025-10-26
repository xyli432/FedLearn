function [train_geo, test_geo, train_t, test_t, train_y, test_y, indices] = spd_split_dataset(geodesic_points, t, y, split_method, param)
    % Function: Split SPD manifold dataset into training and test sets, supporting two splitting strategies
    % Core purpose: Provide separate datasets for model training and evaluation to avoid data leakage
    % Input parameters:
    %   geodesic_points - Set of points on geodesic (dimension: matD×matD×N, N is total number of samples)
    %   t               - Time feature vector (dimension: 1×N)
    %   y               - Set of response variables (dimension: matD×matD×N)
    %   split_method    - Splitting method: 'random' (random split) or 'sequential' (sequential split)
    %   param           - Splitting parameter:
    %                     - For 'random': Test set proportion (e.g., 0.2 means 20% of samples are test set)
    %                     - For 'sequential': Number of training samples (e.g., 80 means first 80 samples are training set)
    % Output parameters:
    %   train_geo, test_geo - Training and test set geodesic points (dimension: matD×matD×number of training/test samples)
    %   train_t, test_t     - Training and test set time features (dimension: 1×number of training/test samples)
    %   train_y, test_y     - Training and test set response variables (dimension: matD×matD×number of training/test samples)
    %   indices             - Split index structure containing:
    %                         - train_idx: Training set sample indices
    %                         - test_idx: Test set sample indices
    
    % -------------------------- Input data consistency check --------------------------
    % Get total number of samples N (from third dimension of geodesic points)
    N = size(geodesic_points, 3);
    % Check: Number of samples in time features and response variables must match geodesic points
    if size(t, 2) ~= N || size(y, 3) ~= N
        error('All input data must have the same number of samples');
    end
    
    % Initialize index structure (to store training and test set sample indices)
    indices = struct();
    
    % -------------------------- Method 1: Random split --------------------------
    if strcmp(split_method, 'random')
        % Check parameter validity: Random split proportion must be between (0,1)
        if param <= 0 || param >= 1
            error('Parameter for random split must be a proportion between (0,1)');
        end
        
        % Calculate number of test samples (total samples × proportion, rounded)
        test_size = round(N * param);
        % Boundary handling: Ensure at least 1 test sample and no more than N-1 (guarantee non-empty training set)
        if test_size == 0
            test_size = 1;
        elseif test_size == N
            test_size = N - 1;
        end
        
        % Randomly select test set indices (without replacement)
        indices.test_idx = randperm(N, test_size);
        % Training set indices = total indices - test set indices (take complement)
        indices.train_idx = setdiff(1:N, indices.test_idx);
        % Sort training set indices (maintain original time order of data)
        indices.train_idx = sort(indices.train_idx);
    
    % -------------------------- Method 2: Sequential split --------------------------
    elseif strcmp(split_method, 'sequential')
        % Check parameter validity: Number of training samples must be an integer between (0,N)
        if param <= 0 || param >= N
            error('Parameter for sequential split must be an integer between (0,N)');
        end
        
        % First param samples as training set, remaining as test set
        indices.train_idx = 1:param;
        indices.test_idx = (param+1):N;
        % Re-check: Ensure test set is not empty
        if isempty(indices.test_idx)
            error('Number of training samples cannot equal total number of samples');
        end
    
    % -------------------------- Handling unsupported splitting methods --------------------------
    else
        error('Splitting method must be ''random'' or ''sequential''');
    end
    
    % -------------------------- Dataset splitting --------------------------
    % Split geodesic points (extract training and test sets by indices)
    train_geo = geodesic_points(:, :, indices.train_idx);
    test_geo = geodesic_points(:, :, indices.test_idx);
    
    % Split time features
    train_t = t(indices.train_idx);
    test_t = t(indices.test_idx);

    % Split response variables
    train_y = y(:, :, indices.train_idx);
    test_y = y(:, :, indices.test_idx);
    
    % (Optional) Display split result information (currently commented out, uncomment to display)
    %fprintf('Dataset splitting completed:\n');
    %fprintf('Number of training samples: %d, Number of test samples: %d\n', ...
    %    length(indices.train_idx), length(indices.test_idx));
end