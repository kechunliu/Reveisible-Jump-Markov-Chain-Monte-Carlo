sigma = [1 -1;-1 4];
mu = [5 10];
corr = -0.5;
iter = 50000;
wave = zeros(2,iter);
coeff = zeros(1,iter);

x = [unifrnd(0,10) unifrnd(5,15)];
wave(:,1)=x;
for i = 2 : iter
    f = mvnpdf(x,mu,sigma);
    %y = normrnd(x,[1 1]);
    y = [5*randn(1) 10*randn(1)];
    f_next = mvnpdf(y,mu,sigma);
    alpha = min(1,f_next/f);
    u = unifrnd(0,1);
    if u < alpha
        x = y;
    end
    wave(:,i) = x';
    cov = corrcoef(wave(1,1:i)',wave(2,1:i)');
    coeff(i) = cov(1,2);
end
plot(coeff,'r');
c1 = mean(coeff(iter*0.98:iter));
hold on;

x = [unifrnd(0,10);unifrnd(5,15)];
wave(:,1)=x;

for i = 2:iter
    x(1) = normrnd(mu(1)+corr*sqrt(sigma(1,1)/sigma(2,2))*(x(2)-mu(2)),sqrt(1-corr^2));
    x(2) = normrnd(mu(2)+corr*sqrt(sigma(2,2)/sigma(1,1))*(x(1)-mu(1)),2*sqrt(1-corr^2));
    
    wave(:,i) = x;
    cov = corrcoef(wave(1,1:i)',wave(2,1:i)');
    coeff(i) = cov(1,2);
end
c = mean(coeff(iter*0.98:iter));

plot(coeff);
legend(strcat('正态分布  ',num2str(c1)),strcat('Gibbs采样  ',num2str(c)));
xlabel('迭代次数');
ylabel('相关系数');

saveas(gcf,'Gibbs+MH.jpg');
