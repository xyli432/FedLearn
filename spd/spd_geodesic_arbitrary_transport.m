function [transported_vecs, projected_points] = spd_geodesic_arbitrary_transport(spd_mfd, geodesic_points, init_tangent_vecs)
    % Function: Perform two core operations on SPD manifold——
    % 1. Parallel transport tangent vectors from the initial point's tangent space to corresponding target points on the geodesic
    % 2. Project transported tangent vectors to SPD manifold via exponential map to generate response points on the manifold
    % Input parameters:
    %   spd_mfd         - SPD manifold object (contains basic manifold operation functions such as mat_to_vec, parallel_transport, etc.)
    %   geodesic_points - Set of points on geodesic, dimension matD×matD×N (matD is SPD matrix dimension, N is number of points)
    %   init_tangent_vecs - Set of tangent vectors in initial point's tangent space, dimension matD×matD×N (one-to-one correspondence with geodesic points)
    % Output parameters:
    %   transported_vecs - Parallel transported tangent vectors, dimension D×N (D=matD×matD, vectorized format)
    %   projected_points - Points after tangent vectors are projected to SPD manifold, dimension matD×matD×N (final response points on manifold)
    
    % -------------------------- Input validity check and parameter initialization --------------------------
    % Get total number of points N on geodesic
    N = size(geodesic_points, 3);
    % Check: Number of tangent vectors must match number of geodesic points (one-to-one transport)
    if size(init_tangent_vecs, 3) ~= N
        error('Number of tangent vectors must match number of geodesic points');
    end
    
    % Determine core dimension parameters
    matD = size(geodesic_points, 1);  % Row and column dimension of SPD matrix (e.g., matD=2 for 2×2 matrix)
    D = matD * matD;                  % Total dimension after vectorization of SPD matrix (e.g., D=4 for 2×2 matrix)
    
    % Extract initial point of geodesic (default to first point on geodesic as starting point for tangent vector transport)
    init_point = geodesic_points(:, :, 1);  % Matrix format of initial point (matD×matD)
    init_point_vec = spd_mfd.mat_to_vec(init_point);  % Convert initial point to vector format (for manifold operations)


    % -------------------------- Initial tangent vector preprocessing: Convert to elements of initial point's tangent space --------------------------
    % Initialize storage: Convert tangent vectors from matrix format to coefficient format in manifold tangent space (dimension: spd_mfd.d × 1 × N)
    % spd_mfd.d is the dimension of SPD manifold's tangent space (equal to matD*(matD+1)/2, i.e., number of independent elements in symmetric matrix)
    init_tangent_vecs_vec = zeros(spd_mfd.d, 1, N);  
    
    % Iterate through each tangent vector to complete format conversion
    for i = 1:N
        current_mat = init_tangent_vecs(:, :, i);  % Matrix format of i-th tangent vector (matD×matD)
        % Extract upper triangular elements of tangent vector matrix (since tangent vectors are symmetric matrices, only need independent elements)
        init_tangent_vecs_vec(:,:,i) = current_mat(triu(true(matD)));
    end
    
    % Convert "coefficient format" of tangent vectors to "element format in initial point's tangent space"
    % (coef_to_log: manifold object method that maps coefficient vector to logarithmic map space of initial point, i.e., tangent space)
    init_tangent_vecs_vec = spd_mfd.coef_to_log(init_point_vec, init_tangent_vecs_vec);
    
    % -------------------------- Tangent vector parallel transport and manifold projection --------------------------
    % Initialize output storage variables
    transported_vecs = zeros(D, N);          % Store transported tangent vectors (vectorized format)
    projected_points = zeros(matD, matD, N);  % Store points projected to manifold (matrix format)
    
    % Iterate through each target point to complete tangent vector transport and projection
    for i = 1:N
        % 1. Get current target point (i-th point on geodesic)
        curr_point = geodesic_points(:, :, i);  % Matrix format
        curr_point_vec = spd_mfd.mat_to_vec(curr_point);  % Convert to vector format (for manifold operations)
        
        % 2. Parallel transport: Transport i-th tangent vector from initial point to tangent space of current target point
        % parallel_transport: manifold object method that implements parallel transport on SPD manifold
        % Input: starting point vector, target point vector, starting point tangent vector; Output: target point tangent vector (vectorized format)
        transported_vecs(:, i) = spd_mfd.parallel_transport(...
            init_point_vec, curr_point_vec, init_tangent_vecs_vec(:, i));
        
        % 3. Exponential map: Project transported tangent vector to SPD manifold to generate point on manifold
        % Exp: manifold object method, exponential map (tangent space→manifold), Input: target point vector, target point tangent vector; Output: manifold point vector
        projected_point_vec = spd_mfd.Exp(curr_point_vec, transported_vecs(:, i));
        % Convert vector format manifold point back to matrix format and store in output variable
        projected_points(:, :, i) = spd_mfd.vec_to_mat(projected_point_vec);
    end
end