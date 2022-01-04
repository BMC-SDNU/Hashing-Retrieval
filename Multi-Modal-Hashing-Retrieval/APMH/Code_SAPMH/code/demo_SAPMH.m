function [] = demo_SAPMH(bits, dataname)
    %myFun - Description
    %
    % Syntax: [] = demo_SAPMH(bits, dataname)
    %
    % Long description

    warning off;
    run = 1;
    bits = str2num(bits);

    fprintf('SAPMH\n');
    %% Contruct Dataset
    fprintf('Contruct dataset...\n')
    [I_te, T_te, L_te, I_db, T_db, L_db, I_tr, T_tr, L_tr, I1, T2, param, L1, L2] = construct_DATASET(dataname);

    %% parameter setting
    % paramether for MIR Flickr
    theta = 1e-4;
    rho = 1e-6;
    lambda = 100;
    n_anchors = 500;
    k = 128;

    % %paramether for NUS-WIDE
    % theta = 1e-7;
    % rho = 1e-5;
    % lambda = 1e-3;
    % n_anchors = 500;
    % k = 500;

    param.bits = bits;
    param.theta = theta;
    param.rho = rho;
    param.lambda = lambda;
    param.n_anchors = n_anchors;
    param.k = k;

    Ntrain = size(I_tr, 1);
    Sc = 2 * (L_tr * L_tr') - ones(Ntrain);
    Ntrain = size(L1, 1);
    S1 = 2 * (L1 * L1') - ones(Ntrain);
    Ntrain = size(L2, 1);
    S2 = 2 * (L2 * L2') - ones(Ntrain);
    L_all = [L_tr; L1; L2];
    Ntrain = size(L_all, 1);
    S = 2 * (L_all * L_all') - ones(Ntrain);

    %% Run algorithm
    for j = 1:run
        fprintf('----------------------------------------------------\n');

        fprintf('Run %d algorithm...\n', j);
        [W, U1, U2, param] = solve_SAPMH(I_tr, T_tr, I1, T2, Sc, S1, S2, S, param);
        fprintf('Algorithm converged!\n')

        %% Evaluation
        fprintf('Evaluation...\n');
        anchorI = param.anchorI;
        anchorT = param.anchorT;
        sigmaI = param.sigmaI;
        sigmaT = param.sigmaT;
        Phi_testI = exp(-sqdist(I_te, anchorI) / (2 * sigmaI * sigmaI));
        Pht_testT = exp(-sqdist(T_te, anchorT) / (2 * sigmaT * sigmaT));
        Phi_dbI = exp(-sqdist(I_db, anchorI) / (2 * sigmaI * sigmaI));
        Pht_dbT = exp(-sqdist(T_db, anchorT) / (2 * sigmaT * sigmaT));

        B_db = online(Phi_dbI, Pht_dbT, W, U1, U2, param);
        B_te = online(Phi_testI, Pht_testT, W, U1, U2, param);
        B_db = compactbit(B_db);
        B_te = compactbit(B_te);

        Dhamm = hammingDist(B_db, B_te);
        [MAP] = perf_metric4Label(L_db, L_te, Dhamm);
        map(j) = MAP;
        fprintf('============================================%d bits SAPMH mAP over %d iterations:%.4f=============================================\n', bits, run, MAP);
    end

    fprintf('[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
end
