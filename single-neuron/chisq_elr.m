function [chi,df,p] = chisq_elr(x)
% does chi-square for contingency tables
% x = table of observations


[r,c]=size(x);
expected = zeros(r,c);

for k=1:c
    sum1(k) = sum(x{:,k},1);
end
for k=1:r
    sum2(k) = sum(x{k,:},2);
end
total = sum(sum1);

for k=1:c
    for j=1:r
        expected(j,k) = (sum1(k)*sum2(j))/total;
    end
end

chi=0;
for k=1:c
    for j=1:r
        chi = chi + (((x{j,k} - expected(j,k))*(x{j,k} - expected(j,k)))/expected(j,k));
    end
end


df = (r-1)*(c-1);
p = 1-chi2cdf(chi,df);

end