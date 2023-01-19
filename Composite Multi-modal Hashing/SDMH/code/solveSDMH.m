function [V_final, W_final, mu1_final, mu2_final] = solveSDMH(phi_x1, phi_x2, Y, param)
%% random initialization
[row, col] = size(phi_x1);
[rowt, ~] = size(phi_x2);
threshold = 0.0001;
lastF = 100000000000000000000000000000;
iter = 1;

beta = param.beta;
delta = param.delta;
gamma = param.gamma;
alpha = param.alpha;
class = param.class;
bits = param.bits;
t = param.t;
%% Initialize
mu1 = 0.5;
mu2 = 0.5;
Phi_X = [sqrt(mu1)*phi_x1', sqrt(mu2)*phi_x2']';
L = size(Phi_X, 1);
V = sign(-1+(1-(-1))*rand(bits, col));
B = sign(-1+(1-(-1))*rand(bits, col));
E = V-B;
%Update U
U = (Phi_X*V') / (V*V'+gamma*eye(bits));
%Update W
W = beta*(V*Phi_X') / (beta*(Phi_X*Phi_X')+gamma*eye(L));
%Update Q
Q = delta*(V*Y') / (delta*(Y*Y')+gamma*eye(class));
%% Iterative algorithm
for iter=1:30
    Phi_X = [sqrt(mu1)*phi_x1', sqrt(mu2)*phi_x2']';
    %Update V    
    V = sign(2*U'*Phi_X - U'*U*B + 2*beta*W*Phi_X + 2*delta*Q*Y + 2*alpha*(B-E/alpha));
    %Update U    
    U = (Phi_X*V') / (V*V'+gamma*eye(bits));
    %Update W
    W = beta*(V*Phi_X') / (beta*(Phi_X*Phi_X')+gamma*eye(L));
    %Update Q
    Q = delta*(V*Y') / (delta*(Y*Y')+gamma*eye(class));
    %Update B
    B = sign(2*alpha*(V+E/alpha)+U'*U*V);
    %Update mu
    U1 = U(1:row, :) / mu1;
    U2 = U(row+1:end, :) / mu2;
    h1 = sum(sum((phi_x1-U1*V).^2)) + gamma*sum(sum((U1).^2));
    h2 = sum(sum((phi_x2-U2*V).^2)) + gamma*sum(sum((U2).^2));
    mu1 = ((1/h1).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1)));
    mu2 = ((1/h2).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1)));
    %Update E
    E = E+alpha*(V-B);
    alpha = alpha*2;
    
    % compute objective function
    norm1 = sum(sum((Phi_X-U*V).^2));
    norm2 = sum(sum((V-W*Phi_X).^2));
    norm3 = sum(sum((V-Q*Y).^2));
    norm4 = sum(sum((U).^2)) + sum(sum((W).^2)) + sum(sum((Q).^2));
    norm5 = sum(sum((V-B+E/alpha).^2));
    currentF = norm1 + beta*norm2 + delta*norm3 + gamma*norm4 + alpha*norm5;
    fprintf('currentF at iteration %d: %.2f; obj: %.4f\n', iter, currentF, lastF - currentF);
    if ((lastF - currentF) < threshold)
        if iter > 3
            return;
        end
    end
    V_final = V;
    W_final = W;
    mu1_final = mu1;
    mu2_final = mu2;
    
    %     iter = iter + 1;
    lastF = currentF;
end
return;

