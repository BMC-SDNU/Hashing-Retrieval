function DMVHparam = trainDMVH(data, param)
%% number of feature types
tic;
train_data = data.train_kdata;

%% parameter setting
threshold = 0.0001;
lastF = 10000000000;
iter = 1;
M = param.M;
alpha = ones(M, 1)/M;
bits = param.nbits;
beta = param.beta;
gamma = param.gamma;
K = 0;
for i = 1:M
        K = K + alpha(i)*train_data.instance{i};
end
[L, N] = size(K);
S = data.S;
D = full(sum(S));
LS = diag(D.^-0.5)*(diag(D)-S)*diag(D.^-0.5);
%% Initialize
R = rand(bits, bits);
Y = sign(-1+(1-(-1))*rand(N, bits));
B = Y;
W = (K*K') \ (K*B);
%% Iterative algorithm
while (true)
    %Update B
    A = LS + gamma*eye(N);
    D = beta*R*R';
    E = -beta*Y*R' - gamma*K'*W;
    B = sylvester(A, D, -E);
    %Update W
    W = (K*K') \ (K*B);
    %Update Y
    Y = sign(B*R);
    %Update R
    [U, ~, V] = svd(B'*Y);
    R = U*V';
    %Update alpha
    E = zeros(M);
    h = zeros(M, 1);
    for i = 1:M
        for j = 1:M
            E(i,j) = 2*trace(train_data.instance{i}'*(W*W')*train_data.instance{i});    
        end
        h(i) = -2*trace(W'*train_data.instance{i}*B) + trace(B'*LS(i)*B);
    end
    E = (E + E');
%   E = E + E' - diag(diag(E));
    Aeq = ones(1,M);
    beq = 1;
    lb = ones(M, 1)*eps;
	ub = ones(M, 1)-lb;
    palpha = alpha;
	opts = optimset('Algorithm','active-set');
	alpha = quadprog(E,h,[],[],Aeq,beq,lb,ub,palpha, opts);
    
    % compute objective function
    norm1 = sum(sum((B'*LS*B).^2));
    norm2 = sum(sum((Y-B*R).^2));
    norm3 = sum(sum((B-K'*W).^2));
    currentF = norm1 + beta*norm2 + gamma*norm3;
    fprintf('\ncurrentF at iteration %d: %.2f; obj: %.4f\n', iter, currentF, lastF - currentF);
    if ((lastF - currentF) < threshold) 
        if iter > 1
            return;
        end
    end
    DMVHparam.Y = Y;
    DMVHparam.B = B;
    DMVHparam.W = W;
    DMVHparam.K = K;
    DMVHparam.alpha = alpha;
       
    iter = iter + 1;
    lastF = currentF;
end
return;