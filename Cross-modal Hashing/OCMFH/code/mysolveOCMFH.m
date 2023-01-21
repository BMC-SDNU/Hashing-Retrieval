function [ WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj] = mysolveOCMFH(Itrain, Ttrain, WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj,lambda, mu, gamma, numiter)
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
bits = size(WI,2);
H = (lambda * WI' * WI + (1- lambda) * WT' * WT + 2 * mu * eye(bits) + gamma * eye(bits)) \ (lambda * WI' * Itrain + (1 - lambda) * WT' * Ttrain + mu * (PI * Itrain + PT * Ttrain));
Uold = [lambda*WI;(1-lambda)*WT];
%% Update Parameters
for i = 1:numiter
    
    % update U1 and U2    
    W1 = W1 + Itrain * H';
    H1 = H1 + H * H';
    
    W2 = W2 + Ttrain * H';
    H2 = H2 + H * H';
    
    WI = W1 / H1;
    WT = W2 / H2;

    
    % update V    
    H = (lambda * WI' * WI + (1- lambda) * WT' * WT + 2 * mu * eye(bits) + gamma * eye(bits)) \ (lambda * WI' * Itrain + (1 - lambda) * WT' * Ttrain + mu * (PI * Itrain + PT * Ttrain));
    
    % update P1 and P2
    F1 = F1 + H*Itrain';   
    G1 = G1 + Itrain*Itrain';

    F2 = F2 + H*Ttrain';
    G2 = G2 + Ttrain*Ttrain';

    PI = F1 / G1;
    PT = F2 / G2;
    
    % compute objective function
    norm1 = lambda * norm(Itrain - WI * H, 'fro');
    norm2 = (1 - lambda) * norm(Ttrain - WT * H, 'fro');
    norm3 = mu * norm(H - PI * Itrain, 'fro');
    norm4 = mu * norm(H - PT * Ttrain, 'fro');
    norm5 = gamma * (norm(WI, 'fro') + norm(WT, 'fro') + norm(H, 'fro') + norm(PI, 'fro') + norm(PT, 'fro'));
    currentF= norm1 + norm2 + norm3 + norm4 + norm5;
    obj = [obj,currentF];
end
    % update HH
    Unew = [lambda*WI;(1-lambda)*WT];
    HH = (Unew' * Unew +  gamma * eye(bits)) \ (Unew' * Uold * HH) ;
    HH = [HH,H];
end
