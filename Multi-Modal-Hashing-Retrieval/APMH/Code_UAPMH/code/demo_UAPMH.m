function [] = demo_UAPMH(bits, dataname)
    %myFun - Description
    %
    % Syntax: [] = demo_UAPMH(bits, dataname)
    %
    % Long description

    warning off;
    run = 1;
    bit = str2num(bits);

    fprintf('UAPMH\n');
    %% Contruct Dataset
    fprintf('Contruct dataset...\n')
    [I_te, T_te, L_te, I_db, T_db, L_db, I_tr, T_tr, L_tr, I1, T2, param, L1, L2] = construct_dataset(dataname);

    %% parameter setting
    theta = 1e-5;
    lambda = 1e-3;
    n_anchors = 500;
    k = 8;

    param.bit = bit;
    param.theta = theta;
    param.lambda = lambda;
    param.n_anchors = n_anchors;
    param.k = k;

    for j = 1:run
        fprintf('----------------------------------------------------\n');
        fprintf('Run %d algorithm...\n', j);
        tic
        [W, U1, U2, param] = solve_UAPMH(I_tr, T_tr, I1, T2, param);
        toc
        fprintf('Algorithm converged!\n')
        %% evaluation
        fprintf('Evaluation...\n');
        tic
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

        % Evaluation
        Dhamm = hammingDist(B_db, B_te);
        [MAP] = perf_metric4Label(L_db, L_te, Dhamm);
        map(j) = MAP;
        fprintf('============================================%d bits UAPMH mAP over %d iterations:%.4f=============================================\n', bit, run, MAP);
    end

    fprintf('[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
end
