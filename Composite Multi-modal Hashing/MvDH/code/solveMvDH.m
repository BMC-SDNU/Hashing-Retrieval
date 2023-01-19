function [B_final, U_final, theta1_final, theta2_final] = solveMvDH(X1, X2, param)
%% random initialization
threshold = 0.0001;
lastF = 100000000000000;
iter = 1;

M = 2;
lambda = 1e-3;
alpha = param.alpha;
beta = param.beta;
gamma = param.gamma;
delta = param.delta;
class = param.class;
bits = param.bits;
%% Initialize
phi_x = [X1', X2'];
Ntrain = size(phi_x,1);
n_anchors = 300;
sample = randsample(Ntrain, n_anchors);
anchor = phi_x(sample,:);

M1=mean(anchor,1);

PP1=anchor- repmat(M1,n_anchors,1);

v1=sum(sum(PP1.^2));

sigma=sqrt(v1/(n_anchors-1));



% sigma = 0.8; % for normalized data  %sigma = mean(mean(D));
% Z construction
nSmp = size(phi_x, 1);
r = 5;
p = 300;
D = EuDist2(phi_x,anchor);
% sigma = mean(mean(D));
% sigma = 120; 
dump = zeros(nSmp,r);
idx = dump;
for i = 1:r
    [dump(:,i),idx(:,i)] = min(D,[],2);
    temp = (idx(:,i)-1)*nSmp+[1:nSmp]';
    D(temp) = 1e100; 
end
dump = exp(-dump/(2*sigma^2));
sumD = sum(dump,2);
Gsdx = bsxfun(@rdivide,dump,sumD);
Gidx = repmat([1:nSmp]',1,r);
Gjdx = idx;
Z=sparse(Gidx(:),Gjdx(:),Gsdx(:),nSmp,p);
jia = diag(Z'*ones(Ntrain,300));
J = diag(jia);
L = Z*(J^-1)*Z';

theta1 = 1/M;
theta2 = 1/M;
theta = [theta1; theta2];
X = [sqrt(theta1)*X1', sqrt(theta2)*X2']'; %dimension*N
[dim, Ntrain] = size(X);
B = sign(-1+(1-(-1))*rand(bits, Ntrain));
opts.p = 300;
F = LSC(X', class, opts); %数据特征一行是一个样本，N*维度
if isvector(F) 
    F = sparse(1:length(F), double(F), 1); F = full(F);
else
    F = F;
end
F = F';
%% Iterative algorithm
while (iter<100)
    % Update U
    U1 = X1*B' / (B*B'+delta*eye(bits));
    U2 = X2*B' / (B*B'+delta*eye(bits));    
    % Update P
    P = (B*B' + delta*eye(bits)) \ (B*F');
    % Update B
    U = [sqrt(theta1)*U1', sqrt(theta2)*U2']';
    X = [sqrt(theta1)*X1', sqrt(theta2)*X2']';
    Q = [X', sqrt(alpha)*F']';
    G = [U; sqrt(alpha)*P'];
    for time = 1:20
        Z0 = B;
        for k = 1:size(B,1)
            Gk = G; Gk(:,k) = [];
            Bk = B; Bk(k,:) = [];
            B(k, :) = sign(G(:,k)'*(Q - Gk*Bk));
        end
        if norm(B-Z0,'fro') < 1e-6 * norm(Z0,'fro')
                    break
        end
    end
    % Update F
    M = L + alpha/beta*(eye(Ntrain) - B'* ((B*B'+delta*eye(bits))^-1) *B);
    FM = F*M;
    if theta>0
        FM = FM + gamma/beta*F*F'*F;
    end
    F = F.*((gamma/beta*F) ./ max(FM,1e-10));
    % Update theta
    h1 = sum(sum((X1 - U1*B).^2));
    h2 = sum(sum((X2 - U2*B).^2));
        if (lambda*(theta1+theta2) + (h2-h1))<=0
            ta1 = 0;
            ta2 = theta1 + theta2;
        else if (lambda*(theta1+theta2) + (h1-h2))<=0
                ta2 = 0;
                ta1 = theta1 + theta2;
            else
                ta1 = (lambda*(theta1+theta2) + (h2-h1)) / (2*lambda);
                ta2 = theta1 + theta2 - ta1;
            end
        end
        theta1 = ta1;
        theta2 = ta2;
        theta = [theta1, theta2]';
        
    % compute objective function
    norm1 = sum(sum((X1-U1*B).^2));
    norm2 = sum(sum((X2-U2*B).^2));
    norm3 = sum(sum((P'*B-F).^2));
    norm4 = trace(F*L*F');
    currentF = theta1*norm1 + theta2*norm2 + alpha*norm3 + beta*norm4;
%     fprintf('\ncurrentF at iteration %d: %.2f; obj: %.4f\n', iter, currentF, lastF - currentF);
    if (lastF-currentF)<threshold
        return
    end
    B_final = B;
    U_final = U;
    theta1_final = theta1;
    theta2_final = theta2;
    
    iter = iter + 1;
    lastF = currentF;
end
