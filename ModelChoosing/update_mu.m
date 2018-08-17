function mu_ = update_mu(par,k)
    mu_ = zeros(k,par.d);
    for i = 1:k
        mu_(i,:) = unifrnd(par.min,par.max);
    end
end