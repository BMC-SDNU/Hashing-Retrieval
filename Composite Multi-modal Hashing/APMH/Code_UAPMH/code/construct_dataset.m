function [I_te, T_te, L_te, I_db, T_db, L_db, I_tr, T_tr, L_tr, I1, T2, param, L1, L2] = construct_dataset_mir(dataname)

    %% Dataset Loading
    addpath('../../../Data');

    if strcmp(dataname, 'flickr')
        load('mir_cnn.mat');
    elseif strcmp(dataname, 'nuswide')
        load('nus_cnn.mat');
    elseif strcmp(dataname, 'coco')
        load('coco_cnn.mat');
    else
        fprintf('ERROR dataname!');
    end

    %% training set
    nn = size(L_tr, 1);
    param.N = nn; % N is the number of traning data
    per = 0;
    per2 = 0.5;
    nc = nn - per * nn;
    param.nc = nc;
    itrain = randperm(length(I_tr), nc);
    I_tr = I_tr(itrain, :);
    T_tr = T_tr(itrain, :);
    L_tr = L_tr(itrain, :);

    param.n1 = per2 * (per * nn);
    itrain = randperm(length(I_tr), param.n1);
    I1 = I_tr(itrain, :);
    L1 = L_tr(itrain, :);

    param.n2 = round((1 - per2) * (per * nn));
    itrain = randperm(length(I_tr), param.n2);
    T2 = T_tr(itrain, :);
    L2 = L_tr(itrain, :);


