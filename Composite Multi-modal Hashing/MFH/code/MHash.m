function [P] = MHash(bits, dataname)
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
    nbits = bits;

    tic
    train_data_sift = T_tr';
    test_data_sift = T_te';
    db_data_sift = T_db';
    train_data_gist = I_tr';
    test_data_gist = I_te';
    db_data_gist = I_db';
    train_labels = L_tr;
    test_labels = L_te;
    db_labels = L_db;

    views{1} = train_data_sift';
    views{2} = train_data_gist';

    % if ~exist(['Laplacian_GK.mat'], 'file')
    para.lamda = 1;
    para.k = 5;
    [L1] = Laplacian_GK(views{1}', para);
    [L2] = Laplacian_GK(views{2}', para);
    save(['Laplacian_GK.mat'], 'L1', 'L2');
    % else
    %     load(['Laplacian_GK.mat']);
    % end

    %% Train the hash functions
    X = [views{1}'; views{2}']; % X is the training data, m*nTr
    [trainNum, trainF] = size(views{1});
    oneline = ones(trainNum, 1);
    gamma = -3; alpha = 0; beta = -2;
    % model_file = sprintf('../%s_%d_%d_%d_%d.mat', model_type, nbits, alpha, beta, gamma);
    % if ~exist(model_file, 'file')
    fprintf('.......Training model.....\n')
    model = multiviewHashing(views, gamma, alpha, beta, L1, L2, nbits);
    %     save(model_file, 'model', '-v7.3');
    % else
    %     fprintf('.......Loading model.....\n')
    %     load(model_file, 'model');
    % end

    toc
    %% Get the hash codes
    tic
    Xdb = [db_data_sift; db_data_gist];
    Xtest = [test_data_sift; test_data_gist];
    dbNum = size(Xdb, 2);
    testNum = size(Xtest, 2);
    onedb = ones(dbNum, 1);
    onetest = ones(testNum, 1);
    Vdb = sign(Xdb' * model.W + onedb * model.b);
    Vtest = sign(Xtest' * model.W + onetest * model.b);
    Dhamm = hammingDist(Vdb + 2, Vtest + 2);
    [P] = perf_metric4Label(db_labels, test_labels, Dhamm);
    toc
end
