function mfd = sphere_manifold()

    %% private interface
    
    D = 3;
    d = 2;
   
    mfd.rtheta = @(theta,phi) [-sin(theta).*sin(phi); ...
              cos(theta).*sin(phi); ...
              zeros(1,length(theta))];
    mfd.rphi = @(theta,phi) [cos(theta).*cos(phi);...
             sin(theta).*cos(phi);...
             -sin(phi)];

    %% public interface
    mfd.d = d; 
    mfd.D = D; 
    mfd.name = 'S2';
    mfd.norm = @mynorm;
    mfd.metric = @metric;
    mfd.co_convert = @co_convert_S2;
    mfd.geodesic = @geodesic;
    mfd.Exp = @Exp;
    mfd.Log = @Log;
    mfd.orthonormal_frame = @orthonormal_frame;
    mfd.dist = @(P,Q) acos(metric(P,Q));
    %mfd.project = @project;
    mfd.parallel_transport = @parallel_transport;
    mfd.coef_process = @coef_process;
    

    %% helper function

    function [R] = mynorm(x,~)
        R = sqrt(metric(x,x));
    end
    
    
    function [M] = metric(U,V,~)
        M = squeeze(U(1,:,:).*V(1,:,:)+U(2,:,:).*V(2,:,:)+U(3,:,:).*V(3,:,:));
  
    end

    function [gd] = geodesic(t,p,h)
        h1 = norm(h,'fro');
        n = size(t,2);
        gd = zeros(3,n);
        for i=1:n
	        gd(:,i) = p * cos(h1*t(i)) + h./h1 * sin(h1*t(i));
        end
    end

    function [q] = Exp(p,v)
	    normu = norm(v,'fro');
        if normu==0
            q = p;
        else
	        q = p * cos(normu) + v./normu * sin(normu);
        end
    end
    
    function [v] = Log(p,q)
	    mm = (q-p) - trace(p'*(q-p)) * p;
        if mm==0
            v=[0;0;0];
        else
	    v = acos(trace(p'*q)) * mm./norm(mm,'fro');
        end
    end

    function [frame] = orthonormal_frame(p)
        e1 = zeros(3,1);
        e2 = zeros(3,1);
        if abs(p(3,1)-1) < 1e-12 || abs(p(3,1)+1) < 1e-12
            e1(:,1) = [1;0;0];
            e2(:,1) = [0;1;0];
        else
            [~,s] = mfd.co_convert(p(:,1));
            e1(:,1) = mfd.rtheta(s(1),s(2));
            e1(:,1) = e1(:,1) ./ mfd.norm(e1(:,1));
            e2(:,1) = mfd.rphi(s(1),s(2));
            e2(:,1) = e2(:,1) ./ mfd.norm(e2(:,1));
        end  
        frame = zeros(3,2);
        frame(:,1) = e1;
        frame(:,2) = e2;
    end

    function [Z] = coef_process(p,V)
        [frame] = orthonormal_frame(p); 
        n = size(V,2);
        d = 2;
        Z = zeros(d,n);
        for j = 1:d
            Z(j,:) = mfd.metric(repmat(frame(:,k,j),1,n),V);
        end
    end

    function [A] = parallel_transport(x1,x2,v1,t)
	    if nargin < 4
		    t=1;
	    end
	    u = Log(x1,x2);
        v = Log(x2,x1);
        if norm(u,'fro')~=0 & norm(v,'fro')~=0
	        e = norm(u,'fro');
	        u = u ./ e;
	        Ac = -x1 * sin(e*t) * u' + u * cos(e*t) * u' + eye(size(u,1)) - u * u';
        else
            Ac = eye(size(x1,1));
        end
        A = Ac*v1;
    end
end