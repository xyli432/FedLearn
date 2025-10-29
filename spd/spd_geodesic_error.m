function mean_error = spd_geodesic_error(spd_mfd, predicted_y, true_y)
    % Function: Calculate geodesic error (average distance) between predicted and true matrices on SPD manifold
    % Core significance: Geodesic distance is the length of the "shortest path" between two points on SPD manifold, more suitable than Euclidean distance for measuring SPD matrix differences
    % Input parameters:
    %   spd_mfd     - SPD manifold object (provides geodesic distance calculation method)
    %   predicted_y - Set of predicted SPD matrices (dimension: matD×matD×N, N is number of samples)
    %   true_y      - Set of true SPD matrices (dimension: matD×matD×N)
    % Output parameters:
    %   mean_error  - Average geodesic error across all samples (scalar)
    
    % Verify input dimension consistency (predicted and true values must have the same dimensions)
    if size(predicted_y) ~= size(true_y)
        error('Predicted and true values must have the same dimensions');
    end
    
    % Get number of samples (read from third dimension)
    N = size(predicted_y, 3);
    errors = zeros(1, N);  % Initialize error storage vector for each sample (1×N)
    
    % Iterate through each sample to calculate geodesic distance
    for i = 1:N
        % Extract predicted and true matrices for i-th sample
        pred_mat = predicted_y(:, :, i);  % Predicted SPD matrix (matD×matD)
        true_mat = true_y(:, :, i);       % True SPD matrix (matD×matD)
        
        % Call distance function of SPD manifold object to calculate geodesic distance between two points
        % Geodesic distance calculation is based on manifold geometric properties, better reflecting true differences in SPD matrices than ordinary Euclidean distance
        errors(i) = spd_mfd.dist(pred_mat, true_mat);
        epsilon = 1e-10 ;
        if any(any(abs(imag(errors(i))) > epsilon))
            errors(i) = (real(errors(i)))^2;
        else
            errors(i) = (real(errors(i)))^2;  
        end 
    end
    
    % Calculate average geodesic error across all samples (as final evaluation metric)
    mean_error = sqrt(mean(errors));  % Average error
end
