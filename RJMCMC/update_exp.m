function [par,D,P] = update_exp(par,D,P,x_train,y_train)
    k_ = par.k;
    d = randi(par.k,1,1);
    mu_ = par.mu;
    mu_(d) = mu_(d)+randn();
    
    D_ = ones(par.N_train,k_);
    for j = 1:k_
        D_(:,j) = mvnpdf(x_train,mu_(j),par.sigma);
    end
    P_ = eye(par.N_train)-D_*pinv(D_'*D_)*D_';
    
    rupdate = ((y_train'*P*y_train)/(y_train'*P_*y_train))^(par.N_train/2);
    
    Aupdate = min(1,rupdate);
    
    u = rand();
    if u < Aupdate
        D = D_;
        P = P_;
        par.k = k_;
        par.mu = mu_;
    end
    
end