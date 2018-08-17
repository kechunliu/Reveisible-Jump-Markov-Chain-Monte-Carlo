function [par,D,P] = death_exp(par,D,P,x_train,y_train)
    k_ = par.k-1;
    d = randi(par.k,1,1);
    temp = par.mu;
    temp(d) = [];
    mu_ = [temp;0];
    
    D_ = ones(par.N_train,k_);
    for j = 1:k_
        D_(:,j) = mvnpdf(x_train,mu_(j),par.sigma);
    end
    P_ = eye(par.N_train)-D_*pinv(D_'*D_)*D_';
    
    rdeath = ((y_train'*P*y_train)/(y_train'*P_*y_train))^(par.N_train/2)*par.k*exp(2)/10;
    
    Adeath = min(1,rdeath);
    
    u = rand();
    if u < Adeath
        D = D_;
        P = P_;
        par.k = k_;
        par.mu = mu_;
    end
    
end