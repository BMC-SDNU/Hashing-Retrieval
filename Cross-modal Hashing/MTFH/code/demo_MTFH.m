function [] = demo_MTFH(q1_bits, q2_bits, dataname)

    data_dir = '../../data';

    if ~exist(data_dir, 'dir')
        error('No such dir(%s)', fullfile(pwd, data_dir))
    end

    if ~exist('../result', 'dir')
        mkdir('../result')
    end

    addpath(data_dir);
    
    if strcmp(dataname, 'flickr')
        load('mir_cnn.mat');
    elseif strcmp(dataname, 'nuswide')
        load('nus_cnn.mat');
    elseif strcmp(dataname, 'coco')
        load('coco_cnn.mat');
    else
        fprintf('ERROR dataname!');
    end 

    addpath(genpath('markSchmidt/'));
    
    %% load wikiData.mat
    
    I_te = bsxfun(@minus, I_te, mean(I_tr, 1));
    I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));
    I_db = bsxfun(@minus, I_db, mean(I_tr, 1));
    
    T_te = bsxfun(@minus, T_te, mean(T_tr, 1));
    T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));
    T_db = bsxfun(@minus, T_db, mean(T_tr, 1));
    

    n1 = size(I_tr, 1);     
    n2 = size(T_tr, 1);     
    n_te = size(I_te, 1);
    n_db = size(I_db, 1)

    %% parameters setting
%     bits = [32, 64]; % show only two cases
    map = zeros(length(q1_bits)* length(q2_bits), 6);
    alpha = 0.5;
    beta = 0.1;
    anchorNum = 500;
    tag = 1;

    %% count Affinity Matrix S in a simple way
    S = L_tr * L_tr';

    for q1 = q1_bits       %
        for q2 = q2_bits   
            %% record q1 and q2. Note that, q1 can be equal or different to q2
            map(tag, 1) = q1;
            map(tag, 2) = q2;

            %% initialization
            U = sign(normrnd(0, 1, n1, q1));
            U_ = sign(normrnd(0, 1, n2, q1));
            V_ = sign(normrnd(0, 1, n1, q2));
            V = sign(normrnd(0, 1, n2, q2));
            H1 = rand(q1, q2);
            H2 = rand(q1, q2);

            %% firt stage
            [U, V, U_, V_, H1, H2] = rndsolveUHV2(alpha, beta, q1, q2, S, U, U_, V, V_, H1, H2);

            %% second stage
            % rnd
            anchorIndex = sort(randperm(n1, anchorNum));
            anchor1 = I_tr(anchorIndex, :);
            anchor2 = T_tr(anchorIndex, :);

            % count Â¦Ã’^2
            z = I_tr * I_tr';
            z = repmat(diag(z), 1, n1)  + repmat(diag(z)', n1, 1) - 2 * z;
            sigma1 = mean(z(:));

            z = T_tr * T_tr';
            z = repmat(diag(z), 1, n1)  + repmat(diag(z)', n1, 1) - 2 * z;
            sigma2 = mean(z(:));

            % kernel logstic regression
            Kanchor1 = kernelMatrix(anchor1, anchor1, sigma1);
            Kanchor2 = kernelMatrix(anchor2, anchor2, sigma2);
            Ktr1 = kernelMatrix(I_tr, anchor1, sigma1);
            Ktr2 = kernelMatrix(T_tr, anchor2, sigma2);
            Kte1 = kernelMatrix(I_te, anchor1, sigma1);
            Kte2 = kernelMatrix(T_te, anchor2, sigma2);
            
            %retrieval databasse
            Kdb1 = kernelMatrix(I_db, anchor1, sigma1);
            Kdb2 = kernelMatrix(T_db, anchor2, sigma2);
            
            HUte1 = zeros(n_te, q1); % image hashing code test
            HVte1 = zeros(n_te, q2); % text hashing code test
            
            HUdb1 = zeros(n_db,q1); % image hashing code database
            HVdb1 = zeros(n_db,q2); % text hasing code database
            
            

            % parameters of KLR
            options = {};
            options.Display = 'final';
            C = 0.01;

            % learn KLR for modality Iamge
            for b = 1: q1
                h = U(:, b);
                funObj = @(u)LogisticLoss(u, Ktr1, h);
                w = minFunc(@penalizedKernelL2, zeros(size(Kanchor1, 1),1), options, Kanchor1, funObj, C);
                HUte1(:, b) = sign(Kte1 * w); %image hashing code test
                
                HUdb1(:, b) = sign(Kdb1 * w); %image hashing code database
            end
            fprintf('======================\n')
            % learn KLR for modality Text
            for b = 1: q2
                h = V(:, b);
                funObj = @(u)LogisticLoss(u, Ktr2, h);
                w = minFunc(@penalizedKernelL2, zeros(size(Kanchor2, 1),1), options, Kanchor2, funObj, C);
                HVte1(:, b) = sign(Kte2 * w);
                
                HVdb1(:, b) = sign(Kdb2 * w);
            end

            % transformation for cross modal retrieval
            HVte2 = sign(HUte1 * H2); % image hashing code test n_te¡Áq2
            HUte2 = sign(HVte1 * H1'); % text hashing code test n_te¡Áq1
            
%             HVdb2 = sign(HUdb1 * H2); % image hashing code database n_db¡Áq2
%             HUdb2 = sign(HVdb1 * H1'); % text hashing code database n_db¡Áq1

            %% map
            U = bitCompact(U >= 0);
            V = bitCompact(V >= 0);

            HUte1 = bitCompact(HUte1 >= 0);
            HVte1 = bitCompact(HVte1 >= 0);
            HUte2 = bitCompact(HUte2 >= 0);
            HVte2 = bitCompact(HVte2 >= 0);
            
            % database
            HUdb1 = bitCompact(HUdb1 >= 0); %image retrieval database
            HVdb1 = bitCompact(HVdb1 >= 0); %image retrieval database
            
%             sim11 = double(hammingDist(HUte1, U))';
%             sim22 = double(hammingDist(HVte1, V))';
            sim12 = double(hammingDist(HVte2, HUdb1))'; %I->T
            sim21 = double(hammingDist(HUte2, HVdb1))'; %T->I
%             map11 = perf_metric4Label(L_tr, L_te, sim11);
%             map22 = perf_metric4Label(L_tr, L_te, sim22);
            map12 = perf_metric4Label(L_db, L_te, sim12);
            map21 = perf_metric4Label(L_db, L_te, sim21);

            %% record result
            % map11: I-->I
            % map22: T-->T
            % map12: I-->T
            % map21: T-->I
%             map(tag, 3) = map11;
%             map(tag, 4) = map22;
            map(tag, 5) = map12;
            map(tag, 6) = map21;
            
            MAP_I2T = map12;
            MAP_T2I = map21;

%             fprintf('result--------------------------\n map11 = %.4f\n map22 = %.4f\n map12 = %.4f\n map21 = %.4f\n', ...
%                 map11, map22, map12, map21);
            
            fprintf('result--------------------------\n map(I->T) = %.4f\n map(T->I) = %.4f\n', ...
                map12, map21);
            
            name = ['../result/' dataname '.txt'];
            fid = fopen(name, 'a+');
            fprintf(fid, '[%s-%d-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, q1_bits, q2_bits, MAP_I2T, MAP_T2I);
            fclose(fid);
            tag = tag + 1;
        end
    end


   


end