function [] = demo_DCMVH(bits, dataname)
    %myFun - Description
    %
    % Syntax: [] = demo_DCMVH(bits, dataname)
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

    %% Parameters setting
    run = 3;
    bits = str2num(bits);
    map = zeros(run, 1);
    theta = 1e-3;
    alpha = 1e-5;
    beta = 1e-5;
    delta = 1e1;
    gamma = 1e1;
    rho = 1e5;

    %% Preparing Data
    fprintf('Preparing data...\n');
    Xa = I_tr';
    Xb = T_tr';
    Xa_test = I_te';
    Xb_test = T_te';
    Xa_db = I_db';
    Xb_db = T_db';

    %% Training & Evaluation Process
    % Seting Parameters
    param.bits = bits;
    param.theta = theta;
    param.alpha = alpha;
    param.beta = beta;
    param.delta = delta;
    param.gamma = gamma;
    param.rho = rho;
    param.Y = L_tr';

    for j = 1:run
        % Training Model
        fprintf('-------------------run %d starts-------------------\n', j);
        [opt] = solveDCMVH(Xa, Xb, param);
        % Out-of-sample Extension
        fprintf('out-of-sample extension...\n');
        mu1 = opt.mu1;
        mu2 = opt.mu2;
        Wa1 = opt.Wa1;
        Wb1 = opt.Wb1;
        Wa2 = opt.Wa2;
        Wb2 = opt.Wb2;
        Wa3 = opt.Wa3;
        Wb3 = opt.Wb3;
        W4 = opt.W4;
        H_te = mu1 * Wa3 * Wa2 * Wa1 * Xa_test + mu2 * Wb3 * Wb2 * Wb1 * Xb_test;
        H_db = mu1 * Wa3 * Wa2 * Wa1 * Xa_db + mu2 * Wb3 * Wb2 * Wb1 * Xb_db;
        B_test = sign(W4 * H_te);
        B_db = sign(W4 * H_db);
        B_db = compactbit(B_db' > 0);
        B_test = compactbit(B_test' > 0);
        % Evaluation
        fprintf('start evaluating...\n');
        Dhamm = hammingDist(B_db + 2, B_test + 2);
        [P2] = perf_metric4Label(L_db, L_te, Dhamm);
        map(j) = P2;
        fprintf('-------------------Run %d Finished!-------------------\n', j);
    end

    fprintf('[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
end
