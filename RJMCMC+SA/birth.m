function [mu_,k_,Pk,Pk_,Dk,Dk_] = birth(par,x,y)
    % propose a new RBF centre at random from the space surrounding x
    centre = zeros(1,par.c);
    for i = 1:par.d
        centre(i) = unifrnd(min(x(:,i))-3,max(x(:,i))+3);
    end
    
    % update parameters
    k_ = par.k+1;
    mu_ = par.mu;
    mu_(k_,:) = centre;
    
    % evaluate Abirth, sample u~U[0,1]
    u = unifrnd(0,1);
    
    Dk = ones(par.N,1+par.d+par.k);
    Dk(:,2:2+par.d-1)=x;
    temp = 1+par.d;
    for i = 1:par.k
        %Dk(:,temp+i)=normpdf(x,par.mu(i)); % exp1
        Dk(:,temp+i)=mvnpdf(x,par.mu(i,:),par.sig); %exp2
    end
    Pk = eye(par.N)-Dk*pinv(Dk'*Dk)*Dk';
    
    Dk_ = ones(par.N,1+par.d+k_);
    Dk_(:,2:2+par.d-1)=x;
    temp = 1+par.d;
    for i = 1:k_
        %Dk_(:,temp+i)=normpdf(x,par.mu(i)); % exp1
        Dk_(:,temp+i)=mvnpdf(x,mu_(i,:),par.sig); % exp2
    end
    Pk_ = eye(par.N)-Dk_*pinv(Dk_'*Dk_)*Dk_';
    
    r = 1;
    for i = 1:par.c
        r = r*(y(:,i)'*Pk*y(:,i))/(y(:,i)'*Pk_*y(:,i));
    end
    rbirth = r^(par.N/2)*par.V*exp(-par.cc)/k_;
    abirth = min(1,rbirth);
    
    % jump or not
    if u > abirth % remain
        mu_ = par.mu;
        k_ = par.k;
        Pk_ = Pk;
        Dk_ = Dk;
    end
end






