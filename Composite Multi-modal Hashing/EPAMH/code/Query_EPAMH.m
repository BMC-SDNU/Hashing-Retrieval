function [Y] = Query_EPAMH(V1, V2, n_iter,R1,R2,mu1,mu2)
%% Online Hashing

threshold = 0.0000001;
lastF = 100000;
[n,bit] = size(V1);
B = randn(n,bit);

for iter=0:n_iter
    currentF = (1/mu1)*sum(sum((B-V1*R1).^2)) + (1/mu2)*sum(sum((B-V2*R2).^2));
    fprintf('Query: currentF at iteration %.7d: %.5f\n ', iter, currentF);
    if (lastF-currentF)<threshold
        if iter >3
            break
        end
    end
    iter = iter + 1;
    lastF = currentF;
    
    % update query B
    Z = ((1/mu1)*V1 * R1)+((1/mu2)*V2 * R2);
    B = ones(size(Z,1),size(Z,2)).*-1;
    B(Z>=0) = 1;
    
    % update query mu
    mu1 = norm( B- V1 * R1, 'fro');
    mu2 = norm( B- V2 * R2, 'fro');
end

% make B binary
Y = zeros(size(B));
Y(B>=0) = 1;
Y = compactbit(Y>0);
