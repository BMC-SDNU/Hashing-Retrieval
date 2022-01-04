function [B] = online(X, Y, W,U1, U2, param)
%% random initializationcq
k=param.k;
[col,row] = size(X);
[colt,rowt] = size(Y);
lambda = param.lambda;
theta = param.theta;
bit=param.bit;
%%initialize B
B = randn(col,bit)>0;B=B*2-1;

%%initialize H
H = randn(col,k);

%initialize mu
mu1 =  1 / (2*sqrt(sum(sum((X - H*U1).^2))));
mu2  = 1 / (2*sqrt(sum(sum((X - H*U2).^2))));

threshold = 0.001;
lastF = 100;
iter = 1;
%% Iterative algorithm
while (iter<10)  
    %Update H
    H = (mu1*X * U1' + mu2*Y * U2' + theta*B * W')/(mu1*U1 * U1' + mu2*U2*U2' + theta*W * W');
    
    %Update B
    B = sign(H * W);
    
    %Update mu
%     mu1 =  1 / (2*sqrt(sum(sum((X - H*U1).^2))));
%     mu2  = 1 / (2*sqrt(sum(sum((X - H*U2).^2))));
mu1=param.mu1;
mu2=param.mu2;
    norm1 = sum(sum((X - H * U1).^2));
    norm2 = sum(sum((Y - H * U2).^2));
    norm3 = sum(sum((B - H * W).^2));
    currentF = mu1*norm1 + mu2*norm2 + theta*norm3;
    fprintf('currentF at iteration %d: %.2f; obj: %.4f\n', iter, currentF, lastF - currentF);      %显示目标函数值
    if (lastF-currentF)<threshold
        if iter >3
            break
        end
    end
    iter = iter + 1;
    lastF = currentF;
end
B=B>0;
end


