function [geodesic_points, t, y] = sphere_generate_outputs(sphere_mfd, N, cov_row, cov_col, hyp_init,theta_params, noise_std, generation_type, start_point, dir_vec)
    % Function: Generate complete dataset on spherical manifold, including geodesic points, time features, and response variables
    % Core process: Validate input legality → generate time features → generate geodesic points → generate tangent vectors → project tangent vectors to response points
    % Input parameters:
    %   sphere_mfd    - Spherical manifold object (provides manifold operation methods such as geodesic generation and norm calculation)
    %   N             - Total number of samples (number of geodesic points and response points to generate)
    %   theta_params  - Parameters for tangent vector generation (length 2, corresponding to 2D tangent space components, used in GP or function generation)
    %   noise_std     - Standard deviation of noise for tangent vectors (controls noise intensity of response variables)
    %   generation_type - Method for generating tangent vectors (string): 'gp' (Gaussian Process) or 'function_plus_noise' (function + noise)
    %   start_point   - Initial point of geodesic (3×1 vector, must be unit vector to ensure it lies on sphere)
    %   dir_vec       - Direction vector of geodesic (3×1 vector, must be in tangent space of start_point, i.e., perpendicular to start_point)
    % Output parameters:
    %   geodesic_points - Set of geodesic points on sphere (3×N matrix, each column vector is a geodesic point)
    %   t               - Time feature vector (1×N row vector, uniformly distributed in [0,1], corresponding to time index of each sample)
    %   y               - Response variables (3×N matrix, points obtained by parallel transport of tangent vectors and projection to sphere via exponential map)
    
    % -------------------------- 1. Validate input parameter legality --------------------------
    % Check 1: Initial point start_point must be unit vector (to ensure it lies on unit sphere)
    % Calculate norm of start_point; if deviation from 1 exceeds 1e-10,判定为非单位向量
    if abs(norm(start_point) - 1) > 1e-10
        error('Input start_point must be a unit vector (lying on the sphere)');
    end
    
    % Check 2: Direction vector dir_vec must be in tangent space of start_point (i.e., perpendicular to start_point)
    % Calculate dot product of the two; if absolute value exceeds 1e-10, there is a radial component and it is not in tangent space
    if abs(dot(start_point, dir_vec)) > 1e-10
        error('Input dir_vec must be perpendicular to start_point (lying in tangent space)');
    end
    
    % -------------------------- 2. Generate time features --------------------------
    % Generate N time points uniformly distributed in [0,1] interval as time index for each sample
    t = linspace(0, 1, N);  % 1×N row vector with time linearly increasing from 0 to 1
    
    % -------------------------- 3. Generate geodesic points on sphere --------------------------
    % 3.1 Normalize direction vector: ensure "movement speed" of geodesic is 1 (prevent direction vector norm from affecting geodesic length)
    dir_norm = sphere_mfd.norm(dir_vec);  % Call manifold object method to calculate norm of dir_vec
    if dir_norm < 1e-10  % Direction vector norm is nearly 0, cannot generate valid geodesic
        error('Direction vector cannot be a zero vector');
    end
    dir_unit = dir_vec / dir_norm;  % Normalized unit direction vector (still in tangent space)

    % 3.2 Generate geodesic points: starting from start_point, along dir_vec direction, generate N geodesic points according to time t
    % Call geodesic method of manifold object, input time t, initial point, direction vector, output 3×N geodesic point matrix
    geodesic_points = sphere_mfd.geodesic(t,start_point,dir_vec);
    
    % -------------------------- 4. Generate tangent vectors and project to response variable y --------------------------
    % 4.1 Generate 2D intrinsic tangent vectors in initial point's tangent space (call defined tangent vector generation function)
    % Input number of samples N, parameters theta_params, noise intensity noise_std, generation method, output 2×N tangent vectors and time features t
    [init_tangent_vecs, t] = sphere_generate_tangent_vectors(N,cov_row, cov_col, hyp_init,theta_params, noise_std, generation_type);

    % 4.2 Parallel transport of tangent vectors + exponential map: generate response points y
    % Call tangent vector transport function to move initial tangent vectors along geodesic to each geodesic point, then project to spherical points
    % In output parameters, ~ ignores transported_vecs (transported tangent vectors), y is projected response points, ~ ignores init_tangent_vecs_3d (3D tangent vectors)
    [~, y,~] = sphere_geodesic_arbitrary_transport(sphere_mfd, geodesic_points, init_tangent_vecs);
end