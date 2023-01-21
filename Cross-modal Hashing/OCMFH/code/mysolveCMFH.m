function [ U1, U2, P1, P2, Y, obj ] = mysolveCMFH( X1, X2, lambda, mu, gamma, numiter, bits)

%% random initialization
[row, col] = size(X1);
rowt = size(X2,1);
Y = rand(bits, col);
U1 = rand(row, bits);
U2 = rand(row, bits);
P1 = rand(bits, row);
P2 = rand(bits, rowt);
threshold = 0.01;
lastF = 99999999;
iter = 1;
obj = [];
%% compute iteratively
while (true)
	% update U1 and U2
    U1 = X1 * Y' / (Y * Y' + gamma * eye(bits));
    U2 = X2 * Y' / (Y * Y' + gamma * eye(bits));
    
	% update Y    
    Y = (lambda * U1' * U1 + (1- lambda) * U2' * U2 + 2 * mu * eye(bits) + gamma * eye(bits)) \ (lambda * U1' * X1 + (1 - lambda) * U2' * X2 + mu * (P1 * X1 + P2 * X2));
    
    %update P1 and P2
    P1 = Y * X1' / (X1 * X1' + gamma * eye(row));
    P2 = Y * X2' / (X2 * X2' + gamma * eye(rowt));
       
    % compute objective function
    norm1 = lambda * norm(X1 - U1 * Y, 'fro');
    norm2 = (1 - lambda) * norm(X2 - U2 * Y, 'fro');
    norm3 = mu * norm(Y - P1 * X1, 'fro');
    norm4 = mu * norm(Y - P2 * X2, 'fro');
    norm5 = gamma * (norm(U1, 'fro') + norm(U2, 'fro') + norm(Y, 'fro') + norm(P1, 'fro') + norm(P2, 'fro'));
    currentF= norm1 + norm2 + norm3 + norm4 + norm5;
    obj = [obj,currentF];
    if (lastF - currentF) < threshold
        return;
    end
    if iter>=numiter
        return
    end
    iter = iter + 1;
    lastF = currentF;
end
return;
end

