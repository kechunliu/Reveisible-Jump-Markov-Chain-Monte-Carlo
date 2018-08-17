clear;clc;

% Data
load '.\mcmc-2017_papers\data2.mat'

x_all = x;
y_all = y;

xtrain = x_all(1:800,:);
ytrain = y_all(1:800,:);
xvalid = x_all(801:1000,:);
yvalid = y_all(801:1000,:);

% Initialization
kmax = 200; %任意选取
iteration = 3000; %length of simulation
par.k = 0; %初始化为0，不需要选取θ
T = 100; %Temperature
error = zeros(iteration,1);
errortest = zeros(iteration,1);
[~,par.d] = size(xtrain); %par.d数据维度
[par.N,par.c] = size(ytrain); %N数据长度，c结果维度
%par.cc = par.c+1; %AIC criteria
par.cc = (par.c+1)*log(par.N)/2; %MDL criteria
par.V = 1;
for i = 1:par.d
    par.V = par.V * (max(xtrain(:,i))-min(xtrain(:,i)));
end
par.mu = zeros(kmax,par.d); %基函数中心点
par.sig = 2*ones(1,par.d); %高斯 方差

% Iteration
for i = 1:iteration
    
    % set the efficients
    if par.k>1 && par.k<kmax
        efficient = 0.2;
    elseif par.k==1
        efficient = 0.25;
    elseif par.k==0
        efficient = 1;
    elseif par.k==kmax
        efficient = 1/3;
    end
    
    % sample u and set the temperature with a cooling schedule
    u = unifrnd(0,1);
    T = T*0.98;
    
    % Move
    if par.k~=kmax && par.k~=0 && par.k~=1
        if u <= efficient 
            [mu_,k_,Pk,Pk_,Dk,Dk_]=birth(par,xtrain,ytrain);
        elseif u <= 2*efficient
            [mu_,k_,Pk,Pk_,Dk,Dk_]=death(par,xtrain,ytrain);
        elseif u <= 3*efficient
            [mu_,k_,Pk,Pk_,Dk,Dk_]=split(par,xtrain,ytrain);
        elseif u <= 4*efficient
            [mu_,k_,Pk,Pk_,Dk,Dk_]=merge(par,xtrain,ytrain);
        else
            [mu_,k_,Pk,Pk_,Dk,Dk_]=update1(par,xtrain,ytrain);
        end
    elseif par.k == 0
        [mu_,k_,Pk,Pk_,Dk,Dk_]=birth(par,xtrain,ytrain);
    elseif par.k == 1
        if u <= efficient
            [mu_,k_,Pk,Pk_,Dk,Dk_]=birth(par,xtrain,ytrain);
        elseif u <= 2*efficient
            [mu_,k_,Pk,Pk_,Dk,Dk_]=death(par,xtrain,ytrain);
        elseif u <= 3*efficient
            [mu_,k_,Pk,Pk_,Dk,Dk_]=split(par,xtrain,ytrain);
        else
            [mu_,k_,Pk,Pk_,Dk,Dk_]=update1(par,xtrain,ytrain);
        end
    end
    
    % Perform an MH step with the annealed acceptance ratio
    % Compute the coefficients alpha
    % Convergence: calculate the error between y and y_
    p = 1;
    p_ = 1;
    for j = 1:par.c
        p = p*ytrain(:,j)'*Pk*ytrain(:,j);
        p_ = p_*ytrain(:,j)'*Pk_*ytrain(:,j);
    end
    p = p^(-par.N/2)*exp(-par.cc*par.k);
    p_ = p_^(-par.N/2)*exp(-par.cc*k_);
    px = p_/p;
    asa = min(1,px^(1/T-1));
    u = unifrnd(0,1);
    if u < asa
        par.mu = mu_;
        par.k = k_;
        alpha = pinv(Dk_'*Dk_)*Dk_'*ytrain;
        y_ = Dk_*alpha;
    else
        alpha = pinv(Dk'*Dk)*Dk'*ytrain;
        y_ = Dk*alpha;
    end
    
    error(i) = norm(ytrain-y_)^2/par.N;
    
    [Ntest,~] = size(xvalid);
    D = ones(Ntest,1+par.d+par.k);
    D(:,2:1+par.d) = xvalid;
    temp = 1+par.d;
    for j = 1:par.k
        D(:,temp+j)=mvnpdf(xvalid,par.mu(j,:),par.sig);
    end
    yvalid_ = D*alpha;
    errortest(i) = norm(yvalid-yvalid_)^2/Ntest; %mse
end

semilogy(error);
hold on;
semilogy(errortest,'r');
legend('训练集','验证集');
saveas(gcf,'error2_log.jpg');

% test_output
[Ntest,ctest] = size(xtest);
D = ones(Ntest,1+ctest+par.k);
D(:,2:2+ctest-1)=xtest;
temp = 1+par.c;
for i = 1:par.k
    D(:,temp+i)=mvnpdf(xtest,par.mu(i,:));
end
ytest = D*alpha;
save('v2.mat','ytest');

save rjmcmc+sa_2.mat

