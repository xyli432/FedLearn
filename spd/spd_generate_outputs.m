function [geodesic_points, t, y] = spd_generate_outputs(spd_mfd, N, matD, cov_row, cov_col, hyp_init, theta_params, start_mat, dir_mat,noise_std, generation_type)
    % Function: Generate complete dataset on SPD manifold, including sample points on geodesic, time features, and response variables
    % Core process: Generate geodesic -> Generate tangent vectors -> Tangent vector transport and projection -> Obtain response variables
    % Input parameters:
    %   spd_mfd         - SPD manifold object (provides basic manifold operation methods)
    %   N               - Number of samples (number of generated points)
    %   theta_params    - Kernel parameters for Gaussian Process (GP) (used to generate tangent vectors)
    %   matD            - Dimension of SPD matrix (e.g., 2 for 2×2 matrix)
    %   start_mat       - Starting point of geodesic (must be an SPD matrix)
    %   dir_mat         - Direction vector of geodesic (must be a symmetric matrix, belonging to tangent space elements)
    %   noise_std       - Noise standard deviation (used to generate noisy tangent vectors)
    %   generation_type - Tangent vector generation method ('gp' or 'function_plus_noise')
    % Output parameters:
    %   geodesic_points - Set of points on geodesic (dimension: matD×matD×N)
    %   t               - Time feature vector (dimension: 1×N, uniformly distributed in [0,1])
    %   y               - Response variables (points where tangent vectors project to manifold, dimension: matD×matD×N)
    
    % -------------------------- Input parameter validity check --------------------------
    % Verify if starting point is an SPD matrix (symmetric positive definite matrix is a valid point on manifold)
    if ~spd_mfd.isspd(start_mat)
        error('Input start_mat must be an SPD matrix');
    end
    % Verify if direction vector is a symmetric matrix (tangent space elements must be symmetric matrices)
    if ~spd_mfd.issym(dir_mat)
        error('Input dir_mat must be a symmetric matrix (tangent space element)');
    end
    
    % -------------------------- Generate time features --------------------------
    % Generate N time points uniformly distributed in [0,1] interval as time features t
    t = linspace(0, 1, N);  % Time uniformly sampled from 0 to 1, total N points
    
    % -------------------------- Generate points on geodesic --------------------------
    % Convert starting point matrix to vector format (manifold operations typically use vector format)
    start_vec = spd_mfd.mat_to_vec(start_mat);
    % Convert direction matrix to vector format
    dir_vec = spd_mfd.mat_to_vec(dir_mat);
    % Generate geodesic: points at time t on manifold starting from start_vec in dir_vec direction (vector format)
    geodesic_vecs = spd_mfd.geodesic(t, start_vec, dir_vec);
    % Convert vector format geodesic points back to matrix format (facilitates subsequent processing)
    geodesic_points = spd_mfd.vec_to_mat(geodesic_vecs);
    
    % -------------------------- Generate tangent vector data --------------------------
    % Call auxiliary function to generate noisy tangent vector data (using GP or function plus noise method)
    % Output: tangent_data (tangent vector matrix set, matD×matD×N), t (time features, consistent with input)
    [tangent_data, t] = spd_generate_gp_variable_dimension(N, matD, cov_row, cov_col, hyp_init, theta_params, noise_std, generation_type);
    % -------------------------- Tangent vector format conversion --------------------------
    % Convert tangent vectors from matD×matD×N matrix format to D×N vector format (D=matD²)
    tangent_vecs = reshape(tangent_data, matD^2, N);
    % Convert vector format back to matrix format (ensure correct dimensions for subsequent transport operations)
    tangent_vecs_mat = spd_mfd.vec_to_mat(tangent_vecs);
    
    % -------------------------- Generate response variable y --------------------------
    % Call function to perform parallel transport and manifold projection of tangent vectors:
    % 1. Transport initial tangent vectors from geodesic starting point to corresponding geodesic points
    % 2. Project transported tangent vectors to SPD manifold via exponential map to obtain response points y
    % Output: ~ (ignore transported tangent vectors), y (projected manifold points, matD×matD×N)
    [~, y] = spd_geodesic_arbitrary_transport(...
        spd_mfd, geodesic_points, tangent_vecs_mat);
end