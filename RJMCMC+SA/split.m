function [mu_,k_,Pk,Pk_,Dk,Dk_] = split(par,x,y)
    % randomly choose an existing RBF centre
    if par.k ~= 1
        d = randi(par.k,1,1);
    else
        d = 1;
    end
    
    % substitute it for two neighbor basis functions, whose centres are
    % obtained using equation(8). The distance between the new bases has to
    % be shorter than the distance between the proposed basis dunction and
    % any other existing basis function.
    
    % update parameters
    k_ = par.k+1;
    ksai = 1;
    ums = rand([1,par.d]);
    mu1 = par.mu(d,:)-ksai*ums;
    mu2 = par.mu(d,:)+ksai*ums;
    mu_ = par.mu;
    mu_(d,:) = mu1;
    mu_(k_,:) = mu2;
    
    % evaluate Asplit, sample u~U[0,1]
    u = unifrnd(0,1);
    
    Dk = ones(par.N,1+par.d+par.k);
    Dk(:,2:2+par.d-1)=x;
    temp = 1+par.d;
    for i = 1:par.k
        %Dk(:,temp+i)=normpdf(x,par.mu(i)); % exp1
        Dk(:,temp+i)=mvnpdf(x,par.mu(i,:),par.sig);
    end
    Pk = eye(par.N)-Dk*pinv(Dk'*Dk)*Dk';
    
    Dk_ = ones(par.N,1+par.d+k_);
    Dk_(:,2:2+par.d-1)=x;
    temp = 1+par.d;
    for i = 1:k_
        %Dk_(:,temp+i)=normpdf(x,par.mu(i)); % exp1
        Dk_(:,temp+i)=mvnpdf(x,mu_(i,:),par.sig);
    end
    Pk_ = eye(par.N)-Dk_*pinv(Dk_'*Dk_)*Dk_';
    
    r = 1;
    for i = 1:par.c
        r = r*(y(:,i)'*Pk*y(:,i))/(y(:,i)'*Pk_*y(:,i));
    end
    rsplit = r^(par.N/2)*par.k*ksai*exp(-par.cc)/(par.k+1);
    asplit = min(1,rsplit);
    
    % jump or not
    if u > asplit % remain
        mu_ = par.mu;
        k_ = par.k;
        Pk_ = Pk;
        Dk_ = Dk;
    end
end






