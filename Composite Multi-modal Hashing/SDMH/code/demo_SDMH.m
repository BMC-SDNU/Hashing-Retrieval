function [] = demo_SDMH(bits, dataname)
    %myFun - Description
    %
    % Syntax: [] = demo_SDMH(bits, dataname)
    %
    % Long description

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

    %% parameter settings
    run = 1;
    map = zeros(run, 1);
    bits = str2num(bits);
    beta = 1e3;
    delta = 1e5;
    gamma = 1e5;
    alpha = 1e-3;
    class = size(L_tr, 2);
    t = 2;

    %% data prepare
    Ntrain = size(I_tr, 1);
    n_anchors = 600;
    sample = randsample(Ntrain, n_anchors);
    anchorI = I_tr(sample, :);
    anchorT = T_tr(sample, :);
    sigmaI = 40;
    sigmaT = 90;
    Phi_dbI = exp(-sqdist(I_db, anchorI) / (2 * sigmaI * sigmaI));
    Phi_dbI = [Phi_dbI, ones(size(Phi_dbI, 1), 1)];
    Phi_dbT = exp(-sqdist(T_db, anchorT) / (2 * sigmaT * sigmaT));
    Phi_dbT = [Phi_dbT, ones(size(Phi_dbT, 1), 1)];
    Phi_testI = exp(-sqdist(I_te, anchorI) / (2 * sigmaI * sigmaI));
    Phi_testI = [Phi_testI, ones(size(Phi_testI, 1), 1)];
    Phi_testT = exp(-sqdist(T_te, anchorT) / (2 * sigmaT * sigmaT));
    Phi_testT = [Phi_testT, ones(size(Phi_testT, 1), 1)];
    Phi_trainI = exp(-sqdist(I_tr, anchorI) / (2 * sigmaI * sigmaI));
    Phi_trainI = [Phi_trainI, ones(size(Phi_trainI, 1), 1)];
    Phi_trainT = exp(-sqdist(T_tr, anchorT) / (2 * sigmaT * sigmaT));
    Phi_trainT = [Phi_trainT, ones(size(Phi_trainT, 1), 1)];
    Phi_dbI = Phi_dbI';
    Phi_dbT = Phi_dbT';
    Phi_trainI = Phi_trainI';
    Phi_trainT = Phi_trainT';
    Phi_testI = Phi_testI';
    Phi_testT = Phi_testT';
    Y = L_tr';

    for i = 1:run
        param.beta = beta;
        param.delta = delta;
        param.gamma = gamma;
        param.alpha = alpha;
        param.class = class;
        param.bits = bits;
        param.t = t;
        %% solve objective function
        [V, W, mu1, mu2] = solveSDMH(Phi_trainI, Phi_trainT, Y, param);
        %% out-of-sample
        Phi_db = [sqrt(mu1) * Phi_dbI', sqrt(mu2) * Phi_dbT']';
        Phi_test = [sqrt(mu1) * Phi_testI', sqrt(mu2) * Phi_testT']';
        Vdb = W * Phi_db;
        Vtest = W * Phi_test;
        %% calculate hash codes
        Vbase = sign((bsxfun(@minus, Vdb', mean(V', 1))));
        Vquery = sign((bsxfun(@minus, Vtest', mean(V', 1))));
        %% evaluate
        Dhamm = hammingDist(Vbase + 2, Vquery + 2);
        [P] = perf_metric4Label(L_db, L_te, Dhamm);
        map(i) = P;
    end

    fprintf('[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
end
