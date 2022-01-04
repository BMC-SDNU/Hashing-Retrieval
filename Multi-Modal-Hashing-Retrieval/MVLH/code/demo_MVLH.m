function [] = demo_MVLH(bits, dataname)
    warning off;
    bits = str2num(bits);
    addpath('../../Data');

    if strcmp(dataname, 'flickr')
        load('mir_cnn.mat');
    elseif strcmp(dataname, 'nuswide')
        load('nus_cnn.mat');
    elseif strcmp(dataname, 'coco')
        load('coco_cnn.mat');
    else
        fprintf('ERROR dataname!');
    end

    run = 1;

    for rrr = 1:run
        lambda = 0.01;
        t = 2;
        %% centralization
        fprintf('centralizing data...\n');
        tic
        Ntrain = size(I_tr, 1);
        % get anchors
        n_anchors = 300;
        sample = randsample(Ntrain, n_anchors);
        anchorI = I_tr(sample, :);
        anchorT = T_tr(sample, :);
        % % determin rbf width sigma
        % Dis = EuDist2(X,anchor,0);
        % % sigma = mean(mean(Dis)).^0.5;
        % sigma = mean(min(Dis,[],2).^0.5);
        % clear Dis
        sigmaI = 85;
        sigmaT = 85;
        Phi_trainI = exp(-sqdist(I_tr, anchorI) / (2 * sigmaI * sigmaI));
        Phi_trainI = [Phi_trainI, ones(size(Phi_trainI, 1), 1)];
        Pht_trainT = exp(-sqdist(T_tr, anchorT) / (2 * sigmaT * sigmaT));
        Pht_trainT = [Pht_trainT, ones(size(Pht_trainT, 1), 1)];
        Phi_trainI = Phi_trainI';
        Pht_trainT = Pht_trainT';
        fprintf('run %d starts...\n', j);
        tic
        I_temp = Phi_trainI;
        T_temp = Pht_trainT;
        param.lambda = lambda;
        param.t = t;
        %% solve objective function
        [U, V, mu1, mu2] = solveMVLH(I_temp, T_temp, bits, param);
        toc
        %% extend to the whole database
        display('Evaluation...');
        tic
        Phi_dbI = exp(-sqdist(I_db, anchorI) / (2 * sigmaI * sigmaI));
        Phi_dbI = [Phi_dbI, ones(size(Phi_dbI, 1), 1)];
        Pht_dbT = exp(-sqdist(T_db, anchorT) / (2 * sigmaT * sigmaT));
        Pht_dbT = [Pht_dbT, ones(size(Pht_dbT, 1), 1)];
        Phi_testI = exp(-sqdist(I_te, anchorI) / (2 * sigmaI * sigmaI));
        Phi_testI = [Phi_testI, ones(size(Phi_testI, 1), 1)];
        Pht_testT = exp(-sqdist(T_te, anchorT) / (2 * sigmaT * sigmaT));
        Pht_testT = [Pht_testT, ones(size(Pht_testT, 1), 1)];
        Phi_dbI = Phi_dbI';
        Pht_dbT = Pht_dbT';
        Phi_testI = Phi_testI';
        Pht_testT = Pht_testT';
        Phi_db = [sqrt(mu1) * Phi_dbI', sqrt(mu2) * Pht_dbT']';
        Vdb = sign(U \ Phi_db);
        Phi_test = [sqrt(mu1) * Phi_testI', sqrt(mu2) * Pht_testT']'; %602*693
        Vtest = sign((U' * U) \ (U' * Phi_test));
        %% calculate hash codes
        Vbase = sign((bsxfun(@minus, Vdb', mean(V', 1))));
        Vquery = sign((bsxfun(@minus, Vtest', mean(V', 1))));
        %% evaluate
        Dhamm = hammingDist(Vbase + 2, Vquery + 2);
        [P] = perf_metric4Label(L_db, L_te, Dhamm);
        map(rrr) = P;
        toc
    end

    fprintf('[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
end
