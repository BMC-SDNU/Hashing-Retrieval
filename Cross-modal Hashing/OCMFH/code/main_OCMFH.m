function [B_Ir,B_Tr,B_Ie,B_Te, PI, PT, HH, obj,traintime,testtime] = main_OCMFH(streamdata, I_te, T_te, bits, lambda, mu, gamma, iter, cmfhiter)
% Reference:
% Di Wang, Quan Wang, Yaqiang An, Xinbo Gao, and Yumin Tian. 
% Online Collective Matrix Factorization Hashing for Large-Scale Cross-Media Retrieval. 
% In 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'20), July 25¨C30, 2020, Virtual Event,
% China. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3397271.3401132
% (Manuscript)
%
% Version1.0 -- Jan/2020
% Contant: Di Wang (wangdi@xidain.edu.cn)
%
if ~exist('lambda','var')
    lambda = 0.5;
end
if ~exist('mu','var')
    mu = 100;
end
if ~exist('gamma','var')
    gamma = 0.001;
end
if ~exist('iter','var')
    iter = 10;
end
if ~exist('cmfhiter','var')
    cmfhiter = 100;
end
%% Training
traintime1 = cputime;
nstream = size(streamdata,2);
% Initialization
Itrain = streamdata{1,1};  Ttrain = streamdata{2,1};
numdata = size(Itrain,1);
mean_I = mean(Itrain, 1);
mean_T = mean(Ttrain, 1);
Itrain = bsxfun(@minus, Itrain, mean_I);
Ttrain = bsxfun(@minus, Ttrain, mean_T);
mean_I = mean_I';
mean_T = mean_T'; 
disp(['Batch:' num2str(1)  ' Total:' num2str(nstream)]);
[WI,WT,PI,PT,HH,obj] = mysolveCMFH(Itrain', Ttrain', lambda, mu, gamma, cmfhiter, bits);

% Training:2--n
mFea1 = size(Itrain,2);
mFea2 = size(Ttrain,2);
W1 = Itrain' * HH';
W2 = Ttrain' * HH';
H1 = HH * HH';
H2 = H1;
F1 = HH*Itrain;
F2 = HH*Ttrain;
G1 = Itrain'*Itrain + gamma*eye(mFea1);
G2 = Ttrain'*Ttrain + gamma*eye(mFea2);

for i = 2:nstream 
    Itrain = streamdata{1,i}';  Ttrain = streamdata{2,i}';
    numdata_tmp = size(Itrain,2);
    mean_Itmp = mean(Itrain, 2);
    mean_Ttmp = mean(Ttrain, 2);
    mean_I = (numdata*mean_I + numdata_tmp*mean_Itmp)/(numdata + numdata_tmp);
    mean_T = (numdata*mean_T + numdata_tmp*mean_Ttmp)/(numdata + numdata_tmp);
    Itrain = bsxfun(@minus, Itrain, mean_I);
    Ttrain = bsxfun(@minus, Ttrain, mean_T);
    numdata = numdata + numdata_tmp;
    disp(['Batch:' num2str(i)  ' Total:' num2str(nstream)]);
    [WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj] = mysolveOCMFH(Itrain, Ttrain, WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj, lambda, mu, gamma, iter);
end
%% Calculate hash codes
Y_tr = sign((bsxfun(@minus, HH , mean(HH,2)))');
Y_tr(Y_tr<0) = 0;
B_Tr = compactbit(Y_tr);
B_Ir = B_Tr;
traintime2 = cputime;
traintime = traintime2 - traintime1;

testtime1 = cputime;
Yi_te = sign((bsxfun(@minus,PI * I_te' , mean(HH,2)))');
Yt_te = sign((bsxfun(@minus,PT * T_te' , mean(HH,2)))');
Yi_te(Yi_te<0) = 0;
Yt_te(Yt_te<0) = 0;
B_Te = compactbit(Yt_te);
B_Ie = compactbit(Yi_te);
testtime2 = cputime;
testtime = testtime2 - testtime1;