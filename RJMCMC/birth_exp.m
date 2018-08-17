function [par,D,P] = birth_exp(par,D,P,x_train,y_train)
    k_ = par.k+1;
    mu_ = par.mu;
    mu_(k_) = 10*rand();
    
    D_ = ones(par.N_train,k_);
    for j = 1:k_
        D_(:,j) = mvnpdf(x_train,mu_(j),par.sigma);
    end
    P_ = eye(par.N_train)-D_*pinv(D_'*D_)*D_';
    
    rbirth = ((y_train'*P*y_train)/(y_train'*P_*y_train))^(par.N_train/2)*10*exp(-2)/k_;
    
    Abirth = min(1,rbirth);
    
    u = rand();
    if u < Abirth || par.k == 0
        D = D_;
        P = P_;
        par.k = k_;
        par.mu = mu_;
    end
    
end