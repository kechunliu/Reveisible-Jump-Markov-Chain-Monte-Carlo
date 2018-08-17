function [mu_,k_,Pk,Pk_,Dk,Dk_] = death(par,x,y)
    % choose the basis centre, to be deleted, at random among the k
    % existing bases.
    if par.k ~= 1
        d = randi(par.k,1,1);
    else
        d = 1;
    end
    
    % update parameters
    k_ = par.k-1;
    temp = par.mu;
    temp(d,:) = [];
    mu_ = [temp;zeros(1,par.d)];
    
    % evaluate Adeath, sample u~U[0,1]
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
    rdeath = r^(par.N/2)*par.k*exp(par.cc)/par.V;
    adeath = min(1,rdeath);
    
    % jump or not
    if u > adeath % remain
        mu_ = par.mu;
        k_ = par.k;
        Pk_ = Pk;
        Dk_ = Dk;
    end
end






