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
clc;clear 
load mirflickr25k.mat
%% Preprocessing data
numbatch = 2000;             
[streamdata,nstream,L_tr,I_tr,T_tr] = predata_stream(I_tr,T_tr,L_tr,numbatch);
%% Calculate the groundtruth
GT = L_te*L_tr';
WtrueTestTraining = zeros(size(L_te,1),size(L_tr,1));
WtrueTestTraining(GT>0)=1;
%% Parameter setting
bit = 32; 
%% Learn OCMFH
[B_I,B_T,tB_I,tB_T,PI,HH] = main_OCMFH(streamdata, I_te, T_te, bit);
%% Compute mAP
Dhamm = hammingDist(tB_I, B_T)';    
[~, HammingRank]=sort(Dhamm,1);
mapIT = map_rank(L_tr,L_te,HammingRank); 
Dhamm = hammingDist(tB_T, B_I)';    
[~, HammingRank]=sort(Dhamm,1);
mapTI = map_rank(L_tr,L_te,HammingRank); 
map = [mapIT(100),mapTI(100)];

fprintf('----------------result------------------\n')
fprintf('map(I->T) = %.4f  map(T->I) = %.4f\n', ...
                map(1), map(2));

