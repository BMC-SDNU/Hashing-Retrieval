function [] = demo_EPAMH(bits, dataname)
    %myFun - Description
    %
    % Syntax: [] = demo_EPAMH(bits, dataname)
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

    %% Training & Evaluation Process
    fprintf('\n============================================Start training EPAMH============================================\n');
    run = 5;
    bit = str2num(bits);

    for j = 1:run

        % PCA X
        XX = I_tr;
        num_training = size(XX, 1);
        sampleMeanX = mean(XX, 1);
        XX = (XX - repmat(sampleMeanX, size(XX, 1), 1));
        [pcx, ~] = eigs(cov(XX(1:num_training, :)), bit);
        XX = XX * pcx;

        %PCA Y
        YY = T_tr;
        num_training = size(YY, 1);
        sampleMeanY = mean(YY, 1);
        YY = (YY - repmat(sampleMeanY, size(YY, 1), 1));
        [pcy, ~] = eigs(cov(YY(1:num_training, :)), bit);
        YY = YY * pcy;

        %% Offline Training
        [R1, R2, mu1, mu2] = solve_EPAMH(XX, YY, 50);

        %% Online Query
        fprintf('start evaluating for query samples...\n');
        %PCA X
        XX = I_te;
        XX = XX * pcx;
        % %PCA Y
        YY = T_te;
        YY = YY * pcy;
        [B_te] = Query_EPAMH(XX, YY, 50, R1, R2, mu1, mu2);

        fprintf('start evaluating for database samples...\n');
        %PCA X
        XX = I_db;
        XX = XX * pcx;
        % %PCA Y
        YY = T_db;
        YY = YY * pcy;
        [B_db] = Query_EPAMH(XX, YY, 50, R1, R2, mu1, mu2);

        % Evaluation
        Dhamm = hammingDist(B_db, B_te);
        [MAP] = perf_metric4Label(L_db, L_te, Dhamm);
        map(j) = MAP;
    end

    fprintf('[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
end
