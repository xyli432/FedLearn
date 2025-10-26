function [transported_vecs, projected_points, init_tangent_vecs_3d] = sphere_geodesic_arbitrary_transport(sphere_mfd, geodesic_points, init_tangent_vecs)
    % Function: Parallel transport 2D tangent vectors from initial point along geodesic to various points on the geodesic, and project to sphere via exponential map
    % Core process: 2D tangent vector → 3D tangent vector → parallel transport along geodesic → projection to sphere
    % Input parameters:
    %   sphere_mfd      - Spherical manifold object (provides orthonormal frame, parallel transport, exponential map, etc.)
    %   geodesic_points - Set of points on geodesic (3×N matrix, each point is 3D spherical coordinate, N is number of points)
    %   init_tangent_vecs - 2D tangent vectors in initial point's tangent space (2×N matrix, each vector corresponds to one sample)
    % Output parameters:
    %   transported_vecs - 3D tangent vectors transported to each geodesic point (3×N matrix)
    %   projected_points - Points projected to sphere via exponential map of tangent vectors (3×N matrix)
    %   init_tangent_vecs_3d - 3D tangent vectors at initial point (3×N matrix, extended from 2D vectors)
    
    % Set numerical precision thresholds (for checking unit vectors, coincident points, etc.)
    eps_norm = 1e-10;          % Threshold for vector norm judgment
    eps_point_eq = 1e-10;      % Threshold for judging coincident point coordinates
    
    % Get number of points N on geodesic and verify input dimensions match
    N = size(geodesic_points, 2);  % Number of samples (obtained from second dimension)
    
    % Check 1: Number of tangent vectors must match number of geodesic points (one tangent vector per point)
    if size(init_tangent_vecs, 2) ~= N
        error('Number of tangent vectors must match number of geodesic points');
    end
    
    % Check 2: Initial tangent vectors must be 2D (sphere is 2D manifold, tangent space is 2D)
    if size(init_tangent_vecs, 1) ~= 2
        error('Initial tangent vectors must be 2D (2×N)');
    end
    
    % Check 3: Geodesic points must be in 3×N spherical point format (3D coordinates)
    if size(geodesic_points, 1) ~= 3 || size(geodesic_points, 2) ~= N
        error('Geodesic points must be 3×N spherical points');
    end
    
    % -------------------------- Step 1: Get initial point and orthonormal frame --------------------------
    % Extract initial point: default to first point on geodesic as starting point for tangent vectors
    init_point = geodesic_points(:, 1);  % 3×1 vector (initial point on sphere)
    
    % Calculate orthonormal frame at initial point (2 orthogonal 3D tangent vectors spanning initial point's tangent space)
    frame = sphere_mfd.orthonormal_frame(init_point);  % 3×2 matrix (columns are orthogonal tangent vectors)
    
    % -------------------------- Step 2: Convert 2D tangent vectors to 3D tangent vectors --------------------------
    % Extend 2D tangent vectors (2×N) at initial point to 3D tangent vectors (3×N) using orthonormal frame
    init_tangent_vecs_3d = zeros(3, N);  % Initialize 3D tangent vector matrix
    for i = 1:N
        a = init_tangent_vecs(1, i);  % First component of 2D tangent vector
        b = init_tangent_vecs(2, i);  % Second component of 2D tangent vector
        
        % Convert 2D vector to 3D tangent vector using orthonormal frame: a×first basis vector + b×second basis vector
        init_tangent_vecs_3d(:,i) = a.*frame(:,1) + b.*frame(:,2);
    end

    % (Comment section: Alternative method for 3D tangent vector conversion in original code, replaced by frame method)
%     x0 = init_point(1);
%     y0 = init_point(2);
%     z0 = init_point(3);
%     
%     % Convert 2D tangent vectors to 3D (within initial point's tangent space)
%     % Tangent space vectors must satisfy perpendicularity with initial point: dot(init_point, vec) = 0
%     init_tangent_vecs_3d = zeros(3, N);  % 3×N
%     
%     for i = 1:N
%         % Get two components of 2D tangent vector
%         a = init_tangent_vecs(1, i);
%         b = init_tangent_vecs(2, i);
%         c = 0;  % Initialize third component
%         
%         % Choose different dimension mapping strategies based on initial point coordinates
%         if abs(z0) > eps_norm  % z component is non-zero, use default mapping [a, b, c]
%             % Calculate c to satisfy perpendicularity constraint: x0*a + y0*b + z0*c = 0
%             c = -(x0*a + y0*b)/z0;
%             init_tangent_vecs_3d(:, i) = [a; b; c];
%             
%         elseif abs(y0) > eps_norm  % y component is non-zero, use mapping [a, c, b]
%             % Calculate c to satisfy perpendicularity constraint: x0*a + y0*c + z0*b = 0
%             c = -(x0*a + z0*b)/y0;
%             init_tangent_vecs_3d(:, i) = [a; c; b];
%             
%         elseif abs(x0) > eps_norm  % x component is non-zero, use mapping [c, a, b]
%             % Calculate c to satisfy perpendicularity constraint: x0*c + y0*a + z0*b = 0
%             c = -(y0*a + z0*b)/x0;
%             init_tangent_vecs_3d(:, i) = [c; a; b];
%             
%         else
%             error('Initial point cannot be (0,0,0) simultaneously, which violates unit sphere definition');
%         end
        % Verify perpendicularity (for debugging)
        % orth_check = dot(init_point, init_tangent_vecs_3d(:, i));
        % if abs(orth_check) > 1e-8
        %     warning('Tangent vector is not perpendicular to initial point, error: %e', orth_check);
        % end
   % end
    
    % -------------------------- Step 3: Parallel transport tangent vectors along geodesic --------------------------
    % Initialize storage variables
    transported_vecs = zeros(3, N);  % Store 3D tangent vectors transported to each point (3×N)
    projected_points = zeros(3, N);  % Store points projected to sphere (3×N)
    
    % Iterate through each geodesic point, perform parallel transport and projection
    for i = 1:N
        % Parallel transport: move 3D tangent vector from initial point to tangent space of i-th geodesic point
        transported_vecs(:, i) = sphere_mfd.parallel_transport(...
            init_point, geodesic_points(:,i), init_tangent_vecs_3d(:, i));
        
        % Exponential map: project transported tangent vector to sphere, obtaining point on sphere
        projected_point = sphere_mfd.Exp(geodesic_points(:,i), transported_vecs(:, i));
        projected_points(:, i) = projected_point;  % Store projection result
    end
end