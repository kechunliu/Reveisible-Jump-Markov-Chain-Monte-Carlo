clear;clc;
% Constant
mu = [5 10];
sigma = [1 -1; -1 4];

% Initialize
iter = 50000;
x = [unifrnd(0,10) unifrnd(5,15)];
coeff = zeros(1,iter);
wave = zeros(2,iter);
wave(:,1) = x';
refuse = zeros(1,3);

% Metropolis-Hastings

% proposal distribution: Gauss
for i = 2 : iter
    f = mvnpdf(x,mu,sigma);
    y = normrnd(x,[1 1]);
    %y = [5*randn(1) 10*randn(1)];
    f_next = mvnpdf(y,mu,sigma);
    alpha = min(1,f_next/f);
    u = unifrnd(0,1);
    if u < alpha
        x = y;
    else
        refuse(1) = refuse(1)+1;
    end
    wave(:,i) = x';
    cov = corrcoef(wave(1,1:i)',wave(2,1:i)');
    coeff(i) = cov(1,2);
end
plot(coeff(1:iter),'r');
c1 = mean(coeff(iter*0.98:iter));
hold on;

% proposal distribution: uniform(0,10) uniform(5,15)
coeff = zeros(1,iter);
for i = 2 : iter
    f = mvnpdf(x,mu,sigma);
    y = unifrnd(x-5,x+5);
    %y = [unifrnd(0,10) unifrnd(5,15)];
    f_next = mvnpdf(y,mu,sigma);
    alpha = min(1,f_next/f);
    u = unifrnd(0,1);
    if u < alpha
        x = y;
    else
        refuse(2) = refuse(2)+1;
    end
    wave(:,i) = x';
    cov = corrcoef(wave(1,1:i)',wave(2,1:i)');
    coeff(i) = cov(1,2);
end
plot(coeff(1:iter),'b');
c2 = mean(coeff(iter*0.98:iter));
% hold on;
% 
% % proposal distribution: exponential lambda=5 lambda=10
% coeff = zeros(1,iter);
% for i = 2 : iter
%     f = mvnpdf(x,mu,sigma);
%     y = [exprnd(5) exprnd(10)];
%     f_next = mvnpdf(y,mu,sigma);
%     alpha = min(1,f_next/f);
%     u = unifrnd(0,1);
%     if u < alpha
%         x = y;
%     else
%         refuse(3) = refuse(3)+1;
%     end
%     wave(:,i) = x';
%     cov = corrcoef(wave(1,1:i)',wave(2,1:i)');
%     coeff(i) = cov(1,2);
% end
% plot(coeff(1:iter),'g');
% c3 = mean(coeff(iter*0.98:iter));

legend(strcat('正态分布  ',num2str(c1)),strcat('均匀分布  ',num2str(c2)));
%legend(strcat('正态分布  ',num2str(c1)),strcat('均匀分布  ',num2str(c2)),strcat('指数分布  ',num2str(c3)));
xlabel('迭代次数');
ylabel('相关系数');

saveas(gcf,'MH-50000_dis.jpg');
