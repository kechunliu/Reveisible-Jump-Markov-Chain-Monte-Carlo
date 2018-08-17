clear;clc;

% Data
x = 10*rand(1000,1);
y = x.^3;

x_train = x(1:800);
y_train = y(1:800);

x_valid = x(801:1000);
y_valid = y(801:1000);

% Parameter
iter = 500;
par.kmax = 100;
par.N_train = 800;
par.N_valid = 200;
par.k = 0;
par.mu = zeros(par.kmax,1);
par.sigma = 1;
D = 0;
P = 0;
error_train = zeros(iter,1);
error_valid = zeros(iter,1);

% Train
for  i = 1:iter
    u = rand();
    
    if par.k>0 && par.k<par.kmax
        efficient = 1/3;
    elseif par.k == par.kmax
        efficient = 1/2;
    end
    
    % Move
    if par.k == 0
        [par,D,P] = birth_exp(par,D,P,x_train,y_train);
    elseif par.k ~= par.kmax
        if u <= efficient
            [par,D,P] = birth_exp(par,D,P,x_train,y_train);
        elseif u <= 2*efficient
            [par,D,P] = death_exp(par,D,P,x_train,y_train);
        else
            [par,D,P] = update_exp(par,D,P,x_train,y_train);
        end
    else
        if u <= efficient
            [par,D,P] = death_exp(par,D,P,x_train,y_train);
        else
            [par,D,P] = update_exp(par,D,P,x_train,y_train);
        end
    end
    
    alpha = pinv(D'*D)*D'*y_train;
    error_train(i) = norm(y_train-D*alpha)^2/800;
    
    % Validation
    Dv = ones(par.N_valid,par.k);
    for j = 1:par.k
        Dv(:,j) = mvnpdf(x_valid,par.mu(j),par.sigma);
    end
    error_valid(i) = norm(y_valid-Dv*alpha)^2/200;
    
end

plot(error_train,'r');hold on;
plot(error_valid,'b');
legend('训练集','验证集');
xlabel('迭代次数');
ylabel('Loss');
saveas(gcf,'RJMCMC_Exp.jpg');



