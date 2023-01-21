function [Wx,Wy,BB,MM] = train0(X, Y, param, L)


%% set the parameters
bits = param.nbits;

alpha = param.alpha;
theta = param.theta;
num_anchor = param.num_anchor;


%% get the dimensions of features
n = size(X,1);  % m
c = size(L,2);

%% initialization
B = sign(randn(n, bits));

P1 = randn(bits, c);


%% iterative optimization
for i = 1:7
    for iter = 1:param.iter
        Ss = randperm(n, num_anchor);

        S = zeros(n,num_anchor);
        S(L * L(Ss,:)' > 0) = 1;
        B = updateB(bits, alpha, theta, B, B(Ss,:), S, P1, L);
    
    end
    P1 = pinv(B'*B)*B'*L;
end

M1 = X'*X;
M2 = X'*B;
M3 = Y'*Y;
M4 = Y'*B;
M5 = B'*B;
M6 = B'*L;
    
    
MM{1,1} = M1;
MM{1,2} = M2;
MM{1,3} = M3;
MM{1,4} = M4;
MM{1,5} = M5;
MM{1,6} = M6;

BB{1,1} = B;
    
    
Wx = pinv(X'*X)*(X'*B);
            
Wy = pinv(Y'*Y)*(Y'*B);

end


function newB = updateB(bits, alpha, theta,newB, newBa, Snn, P1, L)
n = size(newB,1);
na = size(newBa,1);
for k = 1: bits
    THETA = alpha * newB * newBa';
    Cnn = 1 ./ (1+exp(-THETA));
    Bak = newBa(:,k)';
    p = alpha * (Snn-Cnn) .* repmat(Bak,n,1) * ones(na,1) + alpha^2*na/4 * newB(:,k);
        
    vl = 2*L*P1';  
    wl = P1*P1';  
    blk = wl(:,k)'; 
    p2 = vl(:,k) - 2 * newB .* repmat(blk,n,1) * ones(bits,1) + 2 * wl(k,k) * newB(:,k);
  
    
    newB(:,k) = sign(p + theta * p2);
end
end
