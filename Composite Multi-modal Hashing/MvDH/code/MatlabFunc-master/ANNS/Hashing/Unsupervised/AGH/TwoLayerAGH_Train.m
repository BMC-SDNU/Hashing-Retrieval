%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Y, W, Thres, sigma] = TwoLayerAGH_Train(X, Anchor, r, s, sigma, options)
%
% Two-Layer Anchor Graph Hashing Training 
% Written by Wei Liu (wliu@ee.columbia.edu)
% X = nXdim input data 
% Anchor = mXdim anchor points (m<<n)
% r = number of hash bits
% s = number of nearest anchors
% sigma: Gaussian RBF kernel width parameter; if no input (sigma=0) this function will
%        self-estimate and return a value. 
% Y = nXr binary codes (Y_ij in {1,0})
% W = mX(r/2) projection matrix in spectral space
% Thres = 2X(r/2) threshold matrix
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n,dim] = size(X);
m = size(Anchor,1);


%% get Z
Z = zeros(n,m);
Dis = sqdist(X',Anchor');
% clear X;
% clear Anchor;

val = zeros(n,s);
pos = val;
for i = 1:s
    [val(:,i),pos(:,i)] = min(Dis,[],2);
    tep = (pos(:,i)-1)*n+[1:n]';
    Dis(tep) = 1e60; 
end
clear Dis;
clear tep;

if sigma == 0
   sigma = mean(val(:,s).^0.5);
end

switch lower(options.CodingMethod)
    case {lower('Cosine')}
        if isfield(options,'bNormalized') && options.bNormalized
        else
            X = NormalizeFea(X);
        end
        Anchor = NormalizeFea(Anchor);
        S = full(X*Anchor');
        linidx = [1:size(pos,1)]';
        val = S(sub2ind(size(S),linidx(:,ones(1,size(pos,2))),pos));
    case {lower('Gaussian')}
        val = exp(-val/(1/1*sigma^2));
    otherwise
        error('method does not exist!');
end

val = repmat(sum(val,2).^-1,1,s).*val; %% normalize
tep = (pos-1)*n+repmat([1:n]',1,s);
Z([tep]) = [val];
Z = sparse(Z);
clear tep;
clear val;
clear pos;


%% compute eigensystem 
lamda = sum(Z);
M = Z'*Z;
M = diag(lamda.^-0.5)*M*diag(lamda.^-0.5);
[W,V] = eig(full(M));
clear M;
eigenvalue = diag(V)';
clear V;
[eigenvalue,order] = sort(eigenvalue,'descend');
W = W(:,order);
clear order;
ind = find(eigenvalue>0 & eigenvalue<1-1e-3);
eigenvalue = eigenvalue(ind);
W = W(:,ind);
W = diag(lamda.^-0.5)*W(:,1:r/2)*diag(eigenvalue(1:r/2).^-0.5)*sqrt(n);
clear ind;
clear eigenvalue;
clear lambda;


%% get binary codes
Y = Z*W;
Y1 = Y;
Thres = zeros(2,r/2);
for k = 1:r/2
    pos = find(Y(:,k)>0);
    neg = setdiff([1:n],pos);
    n1 = length(pos);
    n2 = n-n1;
    get = Y(:,k);
    get(neg) = -1*get(neg);
    right = full(get'*Z);
    clear get;
    right = right./lamda;
    left = sum(Z(pos,:));
    s1 = sum(left.*right,2);
    left1 = left./lamda;
    s2 = n1-sum(left.*left1,2);
    s3 = sum(Y(pos,k),1);
    clear left
    clear left1;
    clear right;
    
    Thres(1,k) = (2*s3+n2*(s3-s1)/s2)/n;
    Thres(2,k) = (n1*(s3-s1)/s2-2*s3)/n;
    Y1(pos,k) = Y1(pos,k)-Thres(1,k);
    Y1(neg,k) = -Y1(neg,k)+Thres(2,k);
    clear pos;
    clear neg;
end
clear Z;
Y = [Y,Y1];
clear Y1;

Y = (Y>0); %% logical format
