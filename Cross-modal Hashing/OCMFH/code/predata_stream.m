function [streamdata,nstream,L_tr,I_tr,T_tr] = predata_stream(I_tr,T_tr,L_tr,numbatch)
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
[ndata,~] = size(I_tr);
Rdata = randperm(ndata);
I_tr = I_tr(Rdata,:);
T_tr = T_tr(Rdata,:);
L_tr = L_tr(Rdata,:);
nstream = ceil(ndata/numbatch);
streamdata = cell(3,nstream);
for i = 1:nstream-1
    start = (i-1)*numbatch+1;
    endl = i*numbatch;
    streamdata{1,i} = I_tr(start:endl,:);
    streamdata{2,i} = T_tr(start:endl,:);
    streamdata{3,i} = L_tr(start:endl,:);
end
start = (nstream-1)*numbatch+1;
streamdata{1,nstream} = I_tr(start:end,:);
streamdata{2,nstream} = T_tr(start:end,:);
streamdata{3,nstream} = L_tr(start:end,:);

