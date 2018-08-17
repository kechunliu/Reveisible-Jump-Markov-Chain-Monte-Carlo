function [mu_,k_,Pk,Pk_,Dk,Dk_] = merge(par,x,y)
    % choose a basis centre at random among the k existing bases. Then find
    % the closest basis function to it.
    d = randi(par.k,1,1);
    dist = pdist2(par.mu(d,:),par.mu);
    merge = find(dist>min(dist),1);
    
    % if the distance between mu1 and mu2 is less than 2*ksai, substitute
    % the two basis functions for a single basis function in accordance
    % with equation(4.0.3)
    
    % update parameters
    k_ = par.k-1;
    mu_ = par.mu;
    ksai = 1;
    mu1 = par.mu(d,:);
    mu2 = par.mu(merge,:);
    dist = norm(mu1-mu2);
    temp = (mu1+mu2)/2;
    mu_(d,:) = temp;
    mu_(merge,:) = [];
    mu_ =[mu_;zeros(1,par.d)];
    
        % evaluate Amerge, sample u~U[0,1]
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
        rmerge = r^(par.N/2)*par.k/ksai*exp(par.cc)/(par.k-1);
        amerge = min(1,rmerge);
    
        % jump or not
        if u > amerge || dist < 2*ksai % remain
            mu_ = par.mu;
            k_ = par.k;
            Pk_ = Pk;
            Dk_ = Dk;
        end
end






