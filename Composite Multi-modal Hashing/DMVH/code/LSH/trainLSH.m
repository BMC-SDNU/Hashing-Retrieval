function LSHparam = trainLSH(X, LSHparam)
%

[~, Ndim] = size(X);
nbits = LSHparam.nbits;

W = randn(Ndim, nbits);

LSHparam.w = W;