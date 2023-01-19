function [B, U] = compressLSH(X, LSHparam)
%
[Nsamples Ndim] = size(X);
nbits = LSHparam.nbits;  
U = X*LSHparam.w;
B = compactbit(U>0);
U = (U>0);
%[num, ave_num, max_num, min_num]=pcshsta(U>0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cb = compactbit(b)
%
% b = bits array
% cb = compacted string of bits (using words of 'word' bits)

[nSamples nbits] = size(b);
nwords = ceil(nbits/8);
cb = zeros([nSamples nwords], 'uint8');

for j = 1:nbits
    w = ceil(j/8);
    cb(:,w) = bitset(cb(:,w), mod(j-1,8)+1, b(:,j));
end


