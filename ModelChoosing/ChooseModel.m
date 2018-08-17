clear;clc;

% Data
load model1.mat

par.Nmax = 200;

% Initialization
Ms_AIC = zeros(1,par.Nmax);
Ms_BIC = zeros(1,par.Nmax);
Ms_MDL = zeros(1,par.Nmax);
Ms_MAP = zeros(1,par.Nmax);
Ms_HQC = zeros(1,par.Nmax);
plotmax = 2500;

% AIC
for k = 1:par.Nmax
    ksai = k*(par.c+1);
    %Ms_AIC(k) = par.error_valid{k}(iter)^(-par.M/2)*exp(-k*(par.c+1));
    Ms_AIC(k) = -par.M/2*log(par.error_valid{k}(iter))-ksai;
end

% BIC
for k = 1:par.Nmax
    ksai = k*(par.c+1)+par.c*(1+par.d);
    %Ms_BIC(k) = par.error_valid{k}(iter)^(-par.M/2)*par.M^(-ksai/2)*(ksai+2)^(ksai/2+1);
    Ms_BIC(k) = -par.M/2*log(par.error_valid{k}(iter))-ksai/2*log(par.M)-(ksai/2+1)*log(ksai+2);
end

% MDL
for k = 1:par.Nmax
    ksai = k*(par.c+1);
    %Ms_MDL(k) = par.error(k)^(-par.M/2)*par.M^(-ksai/2);
    Ms_MDL(k) = -par.M/2*log(par.error_valid{k}(iter))-ksai/2*log(par.M);
end

% MAP
for k = 1:par.Nmax
    Dv = ones(M,1+d+k);
    Dv(:,2:2+d-1)=x_valid;
    temp = 1+d;
    for j = 1:k
        Dv(:,temp+j)=mvnpdf(x_valid,par.mu{k}(j,:),par.sigma);
    end
    %Ms_MAP(k) = par.error(k)^(-par.M/2)*par.M^(-(ksai^2)/2);
    Ms_MAP(k) = -par.M/2*log(par.error_valid{k}(iter))-1/2*log(det(Dv'*Dv));
end

% HQC
for k = 1:par.Nmax
    ksai = k*(par.c+1);
    %Ms_HQC(k) = par.error(k)^(-par.M/2)*par.M^(-ksai/2);
    Ms_HQC(k) = -par.M/2*log(par.error_valid{k}(iter))-ksai/2*log(log(par.M));
end


% Choose Model
[Mtest,ctest] = size(xtest);

    % AIC
[~,AIC_k] = max(Ms_AIC);

D = ones(par.M,1+par.d+AIC_k);
D(:,2:2+par.d-1)=x_train;
temp = 1+par.d;
for j = 1:AIC_k
    D(:,temp+j)=mvnpdf(x_train,par.mu{AIC_k}(j,:));
end
alpha = pinv(D'*D)*D'*y_train;

D = ones(Mtest,1+ctest+AIC_k);
D(:,2:2+ctest-1)=xtest;
temp = 1+par.d;
for j = 1:AIC_k
    D(:,temp+j)=mvnpdf(xtest,par.mu{AIC_k}(j,:));
end
y1_AIC = D*alpha;

figure;
subplot(3,2,1);
semilogy(par.error{AIC_k}(1:plotmax),'b');
hold on;
semilogy(par.error_valid{AIC_k}(1:plotmax),'r');
% legend('Train','Validation');
title(strcat('AIC k=',num2str(AIC_k)));
xlabel('迭代次数');
ylabel('Loss');
%saveas(gcf,'AIC_2.jpg');

    % BIC
[~,BIC_k] = max(Ms_BIC);

D = ones(par.M,1+par.d+BIC_k);
D(:,2:2+par.d-1)=x_train;
temp = 1+par.d;
for j = 1:BIC_k
    D(:,temp+j)=mvnpdf(x_train,par.mu{BIC_k}(j,:));
end
alpha = pinv(D'*D)*D'*y_train;

D = ones(Mtest,1+ctest+BIC_k);
D(:,2:2+ctest-1)=xtest;
temp = 1+par.d;
for j = 1:BIC_k
    D(:,temp+j)=mvnpdf(xtest,par.mu{BIC_k}(j,:));
end
y1_BIC = D*alpha;
%figure;
subplot(3,2,2);
semilogy(par.error{BIC_k}(1:plotmax),'b');
hold on;
semilogy(par.error_valid{BIC_k}(1:plotmax),'r');
% legend('Train','Validation');
title(strcat('BIC k=',num2str(BIC_k)));
xlabel('迭代次数');
ylabel('Loss');
%saveas(gcf,'BIC_2.jpg');

    % MDL
[~,MDL_k] = max(Ms_MDL);

D = ones(par.M,1+par.d+MDL_k);
D(:,2:2+par.d-1)=x_train;
temp = 1+par.d;
for j = 1:MDL_k
    D(:,temp+j)=mvnpdf(x_train,par.mu{MDL_k}(j,:));
end
alpha = pinv(D'*D)*D'*y_train;

D = ones(Mtest,1+ctest+MDL_k);
D(:,2:2+ctest-1)=xtest;
temp = 1+par.d;
for j = 1:MDL_k
    D(:,temp+j)=mvnpdf(xtest,par.mu{MDL_k}(j,:));
end
y1_MDL = D*alpha;
%figure;
subplot(3,2,3);
semilogy(par.error{MDL_k}(1:plotmax),'b');
hold on;
semilogy(par.error_valid{MDL_k}(1:plotmax),'r');
% legend('Train','Validation');
title(strcat('MDL k=',num2str(MDL_k)));
xlabel('迭代次数');
ylabel('Loss');
%saveas(gcf,'MDL_2.jpg');

    % MAP
[~,MAP_k] = max(Ms_MAP);

D = ones(par.M,1+par.d+MAP_k);
D(:,2:2+par.d-1)=x_train;
temp = 1+par.d;
for j = 1:MAP_k
    D(:,temp+j)=mvnpdf(x_train,par.mu{MAP_k}(j,:));
end
alpha = pinv(D'*D)*D'*y_train;

D = ones(Mtest,1+ctest+MAP_k);
D(:,2:2+ctest-1)=xtest;
temp = 1+par.d;
for j = 1:MAP_k
    D(:,temp+j)=mvnpdf(xtest,par.mu{MAP_k}(j,:));
end
y1_MAP = D*alpha;
%figure;
subplot(3,2,4);
semilogy(par.error{MAP_k}(1:plotmax),'b');
hold on;
semilogy(par.error_valid{MAP_k}(1:plotmax),'r');
% legend('Train','Validation');
title(strcat('MAP k=',num2str(MAP_k)));
xlabel('迭代次数');
ylabel('Loss');
%saveas(gcf,'MAP_2.jpg');


    % HQC
[~,HQC_k] = max(Ms_HQC);

D = ones(par.M,1+par.d+HQC_k);
D(:,2:2+par.d-1)=x_train;
temp = 1+par.d;
for j = 1:HQC_k
    D(:,temp+j)=mvnpdf(x_train,par.mu{HQC_k}(j,:));
end
alpha = pinv(D'*D)*D'*y_train;

D = ones(Mtest,1+ctest+HQC_k);
D(:,2:2+ctest-1)=xtest;
temp = 1+par.d;
for j = 1:HQC_k
    D(:,temp+j)=mvnpdf(xtest,par.mu{HQC_k}(j,:));
end
y1_HQC = D*alpha;
%figure;
subplot(3,2,5);
semilogy(par.error{HQC_k}(1:plotmax),'b');
hold on;
semilogy(par.error_valid{HQC_k}(1:plotmax),'r');
legend('Train','Validation');
title(strcat('HQC k=',num2str(HQC_k)));
xlabel('迭代次数');
ylabel('Loss');
%saveas(gcf,'HQC_2.jpg');



saveas(gcf,'ChooseModel1.jpg');
save('Result1','y1_AIC','y1_BIC','y1_MAP','y1_MDL','y1_HQC');
save 'chooseModel1.mat'







