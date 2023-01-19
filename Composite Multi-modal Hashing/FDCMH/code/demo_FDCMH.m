function [] = demo_FDCMH(bits, dataname)
    %myFun - Description
    %
    % Syntax: [] = demo_FDCMH(bits, dataname)
    %
    % Long description
    warning off;

    %% Dataset Loading
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
    bits = str2num(bits);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    n_anchors = 300;
    param.n_anchors = n_anchors;
    param.t = 2;
    param.eta = 0.001;
    param.rhoo = 2;
    param.alpha = 1e3;
    param.beta = 1e5;
    param.theta = 1e-3;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic
    fprintf('centralizing data...\n');
    Ntrain = size(I_tr, 1);
    S = 2 * L_tr * L_tr' - ones(Ntrain);
    sample = randsample(Ntrain, n_anchors);
    anchorI = I_tr(sample, :);
    anchorT = T_tr(sample, :);

    sigmaI = 85;
    sigmaT = 85;

    PhiI = exp(-sqdist(I_tr, anchorI) / (2 * sigmaI * sigmaI));
    PhiI = [PhiI, ones(Ntrain, 1)];
    PhtT = exp(-sqdist(T_tr, anchorT) / (2 * sigmaT * sigmaT));
    PhtT = [PhtT, ones(Ntrain, 1)];

    S = bits * S;
    map = zeros(run, 1);

    for j = 1:run
        fprintf('run %d starts...\n', j);
        tic
        I_temp = PhiI;
        T_temp = PhtT;
        [W, mu1, mu2, output, wxz] = solveFSMH(I_temp, T_temp, bits, param, S);
        toc

        tic
        Phi_testI = exp(-sqdist(I_te, anchorI) / (2 * sigmaI * sigmaI));
        Phi_testI = [Phi_testI, ones(size(Phi_testI, 1), 1)];
        Pht_testT = exp(-sqdist(T_te, anchorT) / (2 * sigmaT * sigmaT));
        Pht_testT = [Pht_testT, ones(size(Pht_testT, 1), 1)];
        Phi_dbI = exp(-sqdist(I_db, anchorI) / (2 * sigmaI * sigmaI));
        Phi_dbI = [Phi_dbI, ones(size(Phi_dbI, 1), 1)];
        Pht_dbT = exp(-sqdist(T_db, anchorT) / (2 * sigmaT * sigmaT));
        Pht_dbT = [Pht_dbT, ones(size(Pht_dbT, 1), 1)];

        Phi_test = [sqrt(mu1) * Phi_testI, sqrt(mu2) * Pht_testT];
        Phi_db = [sqrt(mu1) * Phi_dbI, sqrt(mu2) * Pht_dbT];
        B_db = Phi_db * W > 0;
        B_test = Phi_test * W > 0;

        B_db = compactbit(B_db);
        B_test = compactbit(B_test);

        fprintf('start evaluating...\n');
        Dhamm = hammingDist(B_db, B_test);
        [P] = perf_metric4Label(L_db, L_te, Dhamm);
        map(j) = P;
        toc
    end

    fprintf('[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
end
