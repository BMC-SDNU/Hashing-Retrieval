function [Wx,Wy,BB,MM] = train(X,Y,param,LChunk, BB, MM,chunki)


%% set the parameters
bits = param.nbits;

alpha = param.alpha;
theta = param.theta;
num_anchor = param.num_anchor;
chunk_size = param.chunk_size;

L = LChunk{chunki,:};

%% get the dimensions of features
n = size(X,1);  % m
c = size(L,2);

%% initialization
B = sign(randn(n, bits));
P1 = randn(bits, c);


%% iterative optimization
for j = 1:7

for iter = 1:param.iter
    
    % update Bx
    Ss = randperm(chunk_size, num_anchor);
    
    oL = randn((chunki-1)*num_anchor, c);
    oB = randn((chunki-1)*num_anchor, bits);
    
    for i = 1:chunki-1
        tL = cell2mat(LChunk(i,:));
        tB = cell2mat(BB(i,1));
        oL((i-1)*num_anchor+1:i*num_anchor,:) = tL(Ss,:);
        oB((i-1)*num_anchor+1:i*num_anchor,:) = tB(Ss,:);
    end

    Sno = zeros(n,(chunki-1)*num_anchor);
    Sno(L * oL' > 0) = 1;
    B = updateB(bits, alpha,theta, B, oB, Sno, P1, L);
    
    BB{chunki,1} = B;

end

P1 = pinv(MM{1,5} + B'*B)*(MM{1,6}+B'*L);

end
 
    % update By
    M1 = X'*X;
    M2 = X'*B;
    M3 = Y'*Y;
    M4 = Y'*B;
    M5 = B'*B;
    M6 = B'*L;
    
    MM{1,1} = MM{1,1} + M1;
    MM{1,2} = MM{1,2} + M2;
    MM{1,3} = MM{1,3} + M3;
    MM{1,4} = MM{1,4} + M4;
    MM{1,5} = MM{1,5} + M5;
    MM{1,6} = MM{1,6} + M6;

    BB{chunki,1} = B;

% % update Wx
Wx = pinv(MM{1,1})*MM{1,2};
    
% update Wy
Wy = pinv(MM{1,3})*MM{1,4};

end
    


function newB = updateB(bits, alpha, theta, newB, oldB, Sno, P1, L)
n = size(newB,1);
on = size(oldB,1);
for k = 1: bits

    PHI = alpha * newB * oldB';
    Cno = 1 ./ (1+exp(-PHI));
    oBk = oldB(:,k)';
    p1 = alpha * (Sno-Cno) .* repmat(oBk,n,1) * ones(on,1) + (on/4 * alpha^2) * newB(:,k);
    
    vl = 2*L*P1';  
    wl = P1*P1';  
    blk = wl(:,k)'; 
    p2 = vl(:,k) - 2 * newB .* repmat(blk,n,1) * ones(bits,1) + 2 * wl(k,k) * newB(:,k);
                
    newB(:,k) = sign(p1 + theta * p2);
end
end

