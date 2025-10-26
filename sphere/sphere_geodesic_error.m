function [mean_error] = sphere_geodesic_error(sphere_mfd, predicted_y, true_y)
    % Function: Calculate geodesic error between predicted points and true points on spherical manifold, output average error
    % Core logic: Based on geodesic distance definition on spherical manifold (length of great circle minor arc), calculate error for each sample and then find the mean
    % Input parameters:
    %   sphere_mfd  - Spherical manifold object (contains built-in method dist for calculating geodesic distance between two points)
    %   predicted_y - Set of predicted spherical points (3×N matrix, 3 is dimension of spherical 3D coordinates, N is number of samples)
    %   true_y      - Set of true spherical points (3×N matrix, must have exactly matching dimensions with predicted_y)
    % Output parameters:
    %   mean_error  - Average geodesic error across all samples (scalar, unit is typically radians, depending on dist method definition)
    % Note: errors (per-sample errors) and std_error (error standard deviation) mentioned in function comments are not outputs; only mean_error is calculated and returned
    
    % -------------------------- Input dimension validation --------------------------
    % Ensure dimensions of predicted and true values match exactly (avoid calculation errors due to mismatched sample count or coordinate dimensions)
    if size(predicted_y) ~= size(true_y)
        error('Dimensions of predicted values and true values must match');  % Throw error and terminate execution if dimensions mismatch
    end
    
    % -------------------------- Initialize parameters --------------------------
    N = size(predicted_y, 2);  % Get number of samples (read from second dimension of matrix, as each sample corresponds to a column)
    errors = zeros(1, N);      % Initialize error array (1×N row vector, stores geodesic error for each sample)
    
    % -------------------------- Calculate geodesic error for each sample --------------------------
    for i = 1:N
        % Extract predicted and true points for current sample (both 3×1 vectors, 3D coordinates on sphere)
        pred_point = predicted_y(:, i);  % Predicted point for i-th sample
        true_point = true_y(:, i);       % True point for i-th sample
        
        % Call dist method of spherical manifold object to calculate geodesic distance between two points
        % Spherical geodesic distance is essentially the "length of great circle minor arc" connecting two points, with unit defined by dist method (typically radians)
        errors(i) = sphere_mfd.dist(pred_point, true_point);
    end
    
    % -------------------------- Calculate average error --------------------------
    % Take arithmetic mean of errors across all samples to get overall average geodesic error
    mean_error = mean(errors);
end