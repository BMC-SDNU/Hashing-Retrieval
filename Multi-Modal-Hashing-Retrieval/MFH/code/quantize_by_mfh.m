function [Y, cbase] = quantize_by_mfh(X, viewsTest, model)
% X is m*nTr, viewsTest is m*nTe
len = model.nbits;
eigvec = model.v;
beta = model.beta;

[~, idx] = sort(model.eigval);
Y = eigvec(:,idx(2:len+1));

% Get Y, W, b
[feaDimX, trainNum] = size(X);
oneline = ones(trainNum,1);
eyemat = eye(trainNum);
eyemat_m = eye(feaDimX);
Lc = eyemat - (oneline*oneline')/trainNum;
W = (X*Lc*X'+beta*eyemat_m)\X*Lc*Y;
b = (oneline'*Y-oneline'*X'*W)/trainNum;

% Start to test
[~, testNum] = size(viewsTest);
oneline = ones(testNum,1);
Y = viewsTest'*W+oneline*b;
median = mean(Y);
cbase=(Y>repmat(median,testNum,1));
% cbase = compactbit(cbase);