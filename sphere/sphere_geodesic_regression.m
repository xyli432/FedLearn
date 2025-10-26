function [train_geo, test_geo, cost] = sphere_geodesic_regression(sphere_mfd,p, v, x, y, test_x, lr, iterations, dim_size)
    % Function: Perform geodesic regression on spherical manifold (optimize parameters via gradient descent to fit geodesic model)
    % Core objective: find geodesic parameters (starting point p, direction v) on spherical manifold that minimize geodesic distance between model outputs and true responses y
    % Input parameters:
    %   sphere_mfd  - Spherical manifold object (provides basic manifold operations; uses custom mapping functions instead of direct internal method calls)
    %   p           - Initial value of geodesic starting point (3×1 vector, point on sphere)
    %   v           - Initial value of geodesic direction (3×1 vector, vector in spherical tangent space)
    %   x           - Training set input features (1×N row vector, e.g., time/index)
    %   y           - Training set true responses (3×N matrix, points on sphere)
    %   test_x      - Test set input features (1×M row vector)
    %   lr          - Learning rate (step size for gradient descent)
    %   iterations  - Number of iterations (total steps for gradient descent)
    %   dim_size    - Sample dimension identifier (used to get sample count, typically 2 for column count)
    % Output parameters:
    %   train_geo   - Geodesic model outputs for training set (3×N matrix, points on sphere)
    %   test_geo    - Geodesic model outputs for test set (3×M matrix, points on sphere)
    %   cost        - Loss value at each iteration (1×iterations vector, mean geodesic distance)

    % Gradient descent iterative optimization of parameters p (starting point) and v (direction)
    for i = 1:iterations
        % -------------------------- 1. Update geodesic starting point p --------------------------
        % Calculate gradient of p (negative gradient direction is parameter update direction)
        LLL = -lr * grad_p(p, v, x, y, dim_size);
        
        % Project gradient to tangent space of p (ensure update direction satisfies spherical manifold constraint: tangent vector perpendicular to p)
        % Principle: LLL = tangential component + radial component (radial component is (LLL'*p)*p), subtract radial component to get pure tangent vector
        if norm(LLL - (LLL' * p) .* p, 2) ~= 0  % Avoid division by zero
            % Normalize tangent vector (maintain stable step size)
            v1 = (norm(LLL, 2) / norm(LLL - (LLL' * p) .* p, 2)) * (LLL - (LLL' * p) .* p);
        else
            v1 = LLL;  % Degenerate case (gradient already in tangent space), use original gradient directly
        end
        
        % Update p via exponential map: map gradient vector v1 in tangent space to new starting point P_new on sphere
        P_new = expmap(v1, p);

        % -------------------------- 2. Update geodesic direction v --------------------------
        % Calculate gradient of v (update in negative gradient direction)
        RRR = v - lr * grad_v(p, v, x, y, dim_size);
        
        % Project gradient to tangent space of p (v is tangent vector at p, must maintain tangent space constraint)
        if norm(RRR - (RRR' * p) .* p, 2) ~= 0  % Avoid division by zero
            v2 = (norm(RRR, 2) / norm(RRR - (RRR' * p) .* p, 2)) * (RRR - (RRR' * p) .* p);
        else
            v2 = RRR;  % Degenerate case, use original gradient directly
        end
        
        % Parallel transport v2: move tangent vector v2 at p to tangent space of new starting point P_new (maintain direction on manifold)
        V_new = transp(v2, p, P_new);

        % -------------------------- 3. Update current parameters and calculate loss --------------------------
        p = P_new;  % Update starting point to new value
        v = V_new;  % Update direction to new value
        
        % Calculate current model outputs for training set (points on geodesic corresponding to x*i, obtained via exponential map to spherical points)
        X_trans = x .* v;  % Tangent vector for each sample (each element of x multiplied by direction v)
        model_out = zeros(size(y));  % Store model outputs
        for j = 1:size(X_trans, 2)
            model_out(:, j) = expmap(X_trans(:, j), P_new);  % Tangent vector → spherical point
        end
        
        % Calculate loss value for current iteration (mean geodesic distance between true y and model outputs)
        cost(i, 1) = compute_cost(y, model_out, dim_size);
    end

    % -------------------------- 4. Generate final model outputs for training and test sets --------------------------
    % Training set: generate geodesic points corresponding to each x using optimized p and v
    train_geo = zeros(3, size(x,2));
    for i = 1:size(x,2)
        train_geo(:, i) = sphere_mfd.Exp(p,v * x(i));  % Call manifold object's exponential map
    end
    
    % Test set: similarly generate geodesic points corresponding to each test_x
    test_geo = zeros(3, size(test_x,2));
    for i = 1:size(test_x,2)
        test_geo(:, i) = sphere_mfd.Exp(p,v * test_x(i));
    end
end

% ------------------------------------------------------------------------------
% Helper function 1: Calculate gradient of geodesic starting point p
% ------------------------------------------------------------------------------
function grad_v_i = grad_v(p, v, x, y, dim_size)
    % Function: Calculate gradient of geodesic direction v (based on error between true y and model outputs)
    % Inputs: p, v, x, y, dim_size as in main function
    % Output: gradient of v (3×1 vector, to be used after projection to tangent space)
    
    grad2 = zeros(size(p));  % Initialize gradient (3×1)
    n = size(y, dim_size);   % Get number of training samples (count by columns)
    
    % Iterate through each sample, accumulate error contributions to v
    for iii = 1:n
        % 1. Model output: geodesic point at x(iii) (p as starting point, v*x(iii) as tangent vector, mapped to sphere via exponential map)
        mapped_value2 = expmap(v * x(iii), p);
        
        % 2. Error vector: logarithmic map from true point y(:,iii) to model output point (spherical error → tangent space vector)
        error_vec2 = logmap(y(:, iii), mapped_value2);
        
        % 3. Parallel transport error vector: move error vector from tangent space of model output point to tangent space of p
        error_vec_transported2 = transp(error_vec2, mapped_value2, p);
        
        % 4. Accumulate gradient: x(iii) weighted error vector (gradient of v is proportional to x)
        grad2 = grad2 + x(iii) * error_vec_transported2;
    end
    
    % Gradient averaging (divide by number of samples) and negation (for loss function minimization direction)
    grad_v_i = -grad2 / n;
end

% ------------------------------------------------------------------------------
% Helper function 2: Calculate gradient of geodesic direction v
% ------------------------------------------------------------------------------
function grad_p_i = grad_p(p, v, x, y, dim_size)
    % Function: Calculate gradient of geodesic starting point p (based on error between true y and model outputs)
    % Inputs: p, v, x, y, dim_size as in main function
    % Output: gradient of p (3×1 vector, to be used after projection to tangent space)
    
    grad1 = zeros(size(p));  % Initialize gradient (3×1)
    n = size(y, dim_size);   % Get number of training samples
    
    % Iterate through each sample, accumulate error contributions to p
    for ii = 1:n
        % 1. Model output: geodesic point at x(ii)
        mapped_value1 = expmap(v * x(ii), p);
        
        % 2. Error vector: logarithmic map from true point to model output point
        error_vec1 = logmap(y(:, ii), mapped_value1);
        
        % 3. Parallel transport error vector: from tangent space of model output point → tangent space of p
        error_vec_transported1 = transp(error_vec1, mapped_value1, p);
        
        % 4. Accumulate gradient (gradient of p is independent of x, directly sum error vectors)
        grad1 = grad1 + error_vec_transported1;
    end
    
    % Gradient averaging and negation
    grad_p_i = -grad1 / n;
end

% ------------------------------------------------------------------------------
% Helper function 3: Calculate geodesic distance between two spherical points
% ------------------------------------------------------------------------------
function dis = geodesic_distance(x, y)
    % Function: Calculate geodesic distance (arc length) between two points x and y on unit sphere
    % Inputs: x, y (both 3×1 vectors, points on sphere)
    % Output: geodesic distance (scalar, unit: radians)
    
    cosine = x' * y;  % Dot product = cos(theta), where theta is central angle
    % Numerical stability handling: ensure dot product is within [-1,1] (avoid errors in acos calculation)
    cosine = max(min(cosine, 1), -1);
    dis = acos(cosine);  % Geodesic distance = central angle (unit sphere has radius 1, so arc length = central angle)
end

% ------------------------------------------------------------------------------
% Helper function 4: Calculate average geodesic loss of the model
% ------------------------------------------------------------------------------
function err = compute_cost(y, y_hat, dim_size)
    % Function: Calculate average geodesic distance between true responses y and model predictions y_hat (loss function)
    % Inputs: y (true values, 3×N), y_hat (predictions, 3×N), dim_size (sample dimension identifier)
    % Output: average loss (scalar)
    
    err = 0;  % Initialize loss
    n = size(y, dim_size);  % Get number of samples
    
    % Iterate through each sample, accumulate geodesic distances
    for i = 1:n
        err = err + geodesic_distance(y(:, i), y_hat(:, i));
    end
    
    % Average loss = total distance / number of samples
    err = err / n;
end

% ------------------------------------------------------------------------------
% Helper function 5: Parallel transport of tangent vectors (direction-preserving vector movement on manifold)
% ------------------------------------------------------------------------------
function u_1 = transp(v, x1, x2)
    % Function: Parallel transport tangent vector v at x1 to tangent space at x2
    % Inputs: v (tangent vector at x1, 3×1), x1 (starting point, 3×1), x2 (target point, 3×1)
    % Output: u_1 (tangent vector at x2, 3×1)
    
    % Calculate logarithmic map from x2 to x1 (tangent vector at x1)
    logmap1 = logmap(x2, x1);
    % Calculate logarithmic map from x1 to x2 (tangent vector at x2)
    logmap2 = logmap(x1, x2);
    % Squared geodesic distance between two points (avoid repeated calculation)
    geo = geodesic_distance(x1, x2)^2;
    
    % Non-degenerate case (points are not coincident, parallel transport can be calculated normally)
    if norm(logmap1, 2) ~= 0 && norm(logmap2, 2) ~= 0 && geo ~= 0
        % Parallel transport formula: remove component of v in logmap1 direction to ensure it lies in x2's tangent space after transport
        u_1 = v - ((logmap1' * v) / geo) * (logmap1 + logmap2);
    else
        u_1 = v;  % Degenerate case (x1=x2), return original vector directly
    end
end

% ------------------------------------------------------------------------------
% Helper function 6: Exponential map of spherical manifold (tangent vector → spherical point)
% ------------------------------------------------------------------------------
function exp_val = expmap(v, p)
    % Function: Map tangent vector v at p to point exp_val on sphere (move ||v|| distance along geodesic)
    % Inputs: v (tangent vector, 3×1), p (starting point, 3×1)
    % Output: exp_val (spherical point, 3×1)
    
    norm_v = norm(v);  % Norm of tangent vector (corresponds to distance moved along geodesic)
    if norm_v < eps  % Degenerate case (movement distance is 0), output starting point p
        exp_val = p;
    else
        % Exponential map formula: p*cos(||v||) + (v/||v||)*sin(||v||)
        exp_val = cos(norm_v) * p + sin(norm_v) * (v / norm_v);
    end
end

% ------------------------------------------------------------------------------
% Helper function 7: Logarithmic map of spherical manifold (spherical point → tangent vector)
% ------------------------------------------------------------------------------
function log_val = logmap(q, p)
    % Function: Map spherical point q to tangent vector log_val at p (vector direction points to q, norm equals geodesic distance between points)
    % Inputs: q (target point, 3×1), p (starting point, 3×1)
    % Output: log_val (tangent vector, 3×1)
    
    cos_theta = p' * q;  % Dot product = cos(theta), where theta is central angle between points
    % Numerical stability handling: ensure dot product is within [-1,1]
    cos_theta = max(min(cos_theta, 1), -1);
    theta = acos(cos_theta);  % Geodesic distance between points (in radians)
    
    if theta < eps  % Degenerate case (p=q), output zero vector
        log_val = zeros(size(p));
    else
        % Logarithmic map formula: theta * (q - p*cos(theta))/sin(theta)
        log_val = theta * (q - cos_theta * p) / sin(theta);
    end
end
%     figure;
%     plot(1:iterations, cost, 'bo-', 'LineWidth', 1, 'MarkerSize', 2);
%     xlabel('Number of Iterations');
%     ylabel('Average Geodesic Distance');
%     title('Convergence Curve of Geodesic Regression');
%     grid on;