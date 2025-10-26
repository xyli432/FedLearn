clear all
N = 100;
N_train = 80;
N_test=20;
start_point = [0; 1; 0];         
dir_vec = [1/sqrt(3); 0; 1/sqrt(4/3)]; 
sphere_mfd = sphere_manifold();
x = linspace(0, 1, N);
geodesic_points = sphere_mfd.geodesic(x,start_point,dir_vec);

cov_row = [1 0;0 1];
hyp_init = log([0.5,1]); 
cov_col= @covSEiso;
generation_type = "gp";
noise_std=0;
y = mv_gptp_sample(cov_col,cov_row,x',hyp_init);
tangent_vectors = zeros(2, N);
y = reshape(y,2,N);
tangent_vectors(1,:) = y(1,:) + noise_std*randn(1,N);
tangent_vectors(2,:) = y(2,:) + noise_std*randn(1,N);

init_point = geodesic_points(:, 1);
frame = sphere_mfd.orthonormal_frame(init_point); 
y1 = zeros(3, N);
for i = 1:N
    a = tangent_vectors(1, i); 
    b = tangent_vectors(2, i);
    y1(:, i) = sphere_mfd.Exp(geodesic_points(:,i), sphere_mfd.parallel_transport(...
    init_point, geodesic_points(:,i), a.*frame(:,1) + b.*frame(:,2)));
end

indices = struct();
indices.test_idx = randperm(N, N_test);
indices.train_idx = setdiff(1:N, indices.test_idx);
train_geo = geodesic_points(:,indices.train_idx);
test_geo = geodesic_points(:,indices.test_idx);
x_train = x(:,indices.train_idx);
x_test = x(:,indices.test_idx);
train_y = y1(:,indices.train_idx);
test_y = y1(:,indices.test_idx);

init_point_train =  train_geo(:,1); 
init_frame = sphere_mfd.orthonormal_frame(init_point_train); 
train_tangent_at_init = zeros(3, N_train); 
train_coords = zeros(2, N_train);
for i = 1:N_train
    geo_point = train_geo(:, i);  
    y_point = train_y(:, i);       
    train_tangent_at_init(:, i) = sphere_mfd.parallel_transport(...
        geo_point, init_point_train,sphere_mfd.Log(geo_point, y_point));
     for j = 1:2
        train_coords(j, i) = sphere_mfd.metric(...
            train_tangent_at_init(:, i), init_frame(:, j));
    end
end

[testL, ~] = gptp_general(x_train',  train_coords', x_test', 0.1, @covSEiso, @SE_init, 'All');
test_coords = reshape(testL.mean',2,N_test);

RMSE_mgp1 = sqrt(mse(test_coords(:,1), tangent_vectors(1,indices.test_idx)'));
RMSE_mgp2 = sqrt(mse(test_coords(:,2), tangent_vectors(2,indices.test_idx)'));

test_tangent_at_init = zeros(3, N_test);  
predicted_y = zeros(3,N_test);
for i = 1:N_test
    geo_point = test_geo(:, i); 
    for j = 1:2
        test_tangent_at_init(:, i) = test_tangent_at_init(:, i) + ...
            test_coords(j, i) * init_frame(:, j);
    end
    predicted_y(:, i) = sphere_mfd.Exp(geo_point, sphere_mfd.parallel_transport(...
            init_point_train, geo_point, test_tangent_at_init(:, i)));
end


sphere_geodesic_error(sphere_mfd, predicted_y, test_y)





















RMSE_mgp1 = sqrt(mse(testL.mean(:,1), tangent_vectors(1,indices.test_idx)'));
RMSE_mgp2 = sqrt(mse(testL.mean(:,2), tangent_vectors(2,indices.test_idx)'));


















