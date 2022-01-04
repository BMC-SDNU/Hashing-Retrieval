function [] = demo_MvDH(bits, dataname)

    warning off;
    run = 1;

    addpath('../../Data');
    addpath('MatlabFunc-master');
    addpath(genpath(pwd));

    if strcmp(dataname, 'flickr')
        load('mir_cnn.mat');
    elseif strcmp(dataname, 'nuswide')
        load('nus_cnn.mat');
    elseif strcmp(dataname, 'coco')
        load('coco_cnn.mat');
    else
        fprintf('ERROR dataname!');
    end
    
    for rrr = 1:run

        %% parameter settings
        bits = str2num(bits);
        alpha = 1e-3;
        beta = 1e-3;
        gamma = 1e6;
        delta = 1e-2;
        class = 24;
        %% centralization
        fprintf('centralizing data...\n');
        I_te = bsxfun(@minus, I_te', mean(I_tr', 2));
        I_tr = bsxfun(@minus, I_tr', mean(I_tr', 2));
        T_te = bsxfun(@minus, T_te', mean(T_tr', 2));
        T_tr = bsxfun(@minus, T_tr', mean(T_tr', 2));

        %% training model

        fprintf('\nrun %d starts...', rrr);
        tic
        param.alpha = alpha;
        param.beta = beta;
        param.gamma = gamma;
        param.delta = delta;
        param.class = class;
        param.bits = bits;
        %% solve objective function
        [B, U, theta1, theta2] = solveMvDH(I_tr, T_tr, param);
        toc
        %% extend to the whole database
        tic
        Xdb = [sqrt(theta2) * I_db, sqrt(theta2) * T_db];
        Ndb = size(Xdb, 1);
        B_db = sign(-1 + (1 - (-1)) * rand(bits, Ndb)); %16*5482

        for time = 1:10
            Z0 = B_db;

            for k = 1:size(B_db, 1)
                Ukk = U; Ukk(:, k) = [];
                Bkk = B_db; Bkk(k, :) = [];
                B_db(k, :) = sign(U(:, k)' * (Xdb' - Ukk * Bkk));
            end

            if norm(B_db - Z0, 'fro') < 1e-6 * norm(Z0, 'fro')
                break
            end

        end

        Xtst = [sqrt(theta2) * I_te', sqrt(theta2) * T_te'];
        Ntest = size(Xtst, 1);
        B_tst = sign(-1 + (1 - (-1)) * rand(bits, Ntest));

        for time = 1:10
            Z0 = B_tst;

            for k = 1:size(B_tst, 1)
                Ukk = U; Ukk(:, k) = [];
                Bkk = B_tst; Bkk(k, :) = [];
                B_tst(k, :) = sign(U(:, k)' * (Xtst' - Ukk * Bkk));
            end

            if norm(B_tst - Z0, 'fro') < 1e-6 * norm(Z0, 'fro')
                break
            end

        end

        %% calculate hash codes
        B_base = sign((bsxfun(@minus, B_db', mean(B', 1))));
        B_tst = sign((bsxfun(@minus, B_tst', mean(B', 1))));

        %% evaluate
        fprintf('start evaluating...\n');
        Dhamm = hammingDist(B_base + 2, B_tst + 2);
        [P] = perf_metric4Label(L_db, L_te, Dhamm);
        map(rrr) = P;
        toc
    end

    fprintf('[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
end
