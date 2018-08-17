clear;clc;

%Data
load 'data1.mat'

x_all = x;
y_all = y;

x_train = x_all(1:800,:);
y_train = y_all(1:800,:);

x_valid = x_all(801:1000,:);
y_valid = y_all(801:1000,:);


% Parameter
iter = 5000;
par.Nmax = 200;
[par.M,par.d] = size(x_train);
[~,par.c] = size(y_train);
par.mu = cell(1,par.Nmax);
par.error = cell(1,par.Nmax);
par.error_valid = cell(1,par.Nmax);
par.min = min(x_train);
par.max = max(x_train);
par.sigma = 2*ones(1,par.d);

% Iteration
for k = 1:par.Nmax
    par.error{k} = zeros(1,iter);
    
    if mod(k,2) == 0
        k
    end;
    
    % Train
    for i = 1:iter
        
        % proposal distribution: uniform
        mu_ = update_mu(par,k);
        
        D = ones(par.M,1+par.d+k);
        D(:,2:2+par.d-1)=x_train;
        temp = 1+par.d;
        for j = 1:k
            D(:,temp+j)=mvnpdf(x_train,mu_(j,:),par.sigma);
        end
        P = eye(par.M)-D*pinv(D'*D)*D';
        alpha_ = pinv(D'*D)*D'*y_train;
        
        sigma2 = norm(y_train-D*alpha_)^2/800;
        
        if i == 1
            par.error{k}(i) = sigma2;
            par.mu{k} = mu_;
            alpha = alpha_;
        elseif sigma2 < par.error{k}(i-1)
            par.error{k}(i) = sigma2;
            par.mu{k} = mu_;
            alpha = alpha_;
        else
            par.error{k}(i) = par.error{k}(i-1);
        end
        
        % Validation
        [M,d] = size(x_valid);
        Dv = ones(M,1+d+k);
        Dv(:,2:2+d-1)=x_valid;
        temp = 1+d;
        for j = 1:k
            Dv(:,temp+j)=mvnpdf(x_valid,par.mu{k}(j,:),par.sigma);
        end
        Pv = eye(M)-Dv*pinv(Dv'*Dv)*Dv';
        yvalid_ = Dv*alpha;
        par.error_valid{k}(i) = norm(yvalid_-y_valid)^2/M;
        
    end
    
end

% Save data
save model1.mat



