close all; clear; clc;

% nbits_set=[8 16 32 64 128];
nbits_set = [8];

%% load dataset

set = 'MIRFlickr';



%% initialization
fprintf('initializing...\n')
alphas = [0.45 0.35 0.25 0.15 0.1];
param.datasets = set;
if strcmp(set,'MIRFlickr')
    load('MIRFLICKR.mat');
    param.iter = 3; 
    param.num_anchor = 50;
    param.theta = 0.1;
    param.chunk_size = 2000;
    X = XAll; Y = YAll; L = LAll;
    R = randperm(size(L,1));
    queryInds = R(1:2000);
    sampleInds = R(2001:end);
    param.nchunks = floor(length(sampleInds)/param.chunk_size);
       
    XChunk = cell(param.nchunks,1);
    YChunk = cell(param.nchunks,1);
    LChunk = cell(param.nchunks,1);
    for subi = 1:param.nchunks-1
        XChunk{subi,1} = X(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
        YChunk{subi,1} = Y(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
        LChunk{subi,1} = L(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
    end
    XChunk{param.nchunks,1} = X(sampleInds(param.chunk_size*subi+1:end),:);
    YChunk{param.nchunks,1} = Y(sampleInds(param.chunk_size*subi+1:end),:);
    LChunk{param.nchunks,1} = L(sampleInds(param.chunk_size*subi+1:end),:);
        
    XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
    clear X Y L
end


for bit=1:length(nbits_set) 
    nbits = nbits_set(bit);
    param.alpha = alphas(bit);
    
    %% DOCH
    param.nbits=nbits;
    evaluate(XChunk,YChunk,LChunk,XTest,YTest,LTest,param);

end
