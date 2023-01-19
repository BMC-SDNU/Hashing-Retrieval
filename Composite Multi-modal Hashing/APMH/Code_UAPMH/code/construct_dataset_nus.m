function [I_te, T_te, L_te, I_db, T_db, L_db, I_tr, T_tr, L_tr, I1, T2, param, L1, L2] = construct_dataset_nus(~)
    load('nus_cnn.mat')
    img = [I_db; I_te];
    txt = [T_db; T_te];
    LAll = [L_db; L_te];

    N = size(LAll, 1);
    index = [1:N];
    index = index';

    for i = 1:21 %24 is the number of categories
        [row{i}, ~] = find(LAll(:, i) == 1);
        n_size = size(row{i}, 1);
        R = randperm(n_size, 100); %select 100 images for each category
        indQ{i} = row{i}(R, :);
    end

    j = 1; n = 100;

    for i = 1:21
        ind(j:n, :) = indQ{1, i}(:, :);
        j = j + 100;
        n = n + 100;
    end

    %% query set
    indQ = unique(ind);
    indexQuery = indQ;
    I_te = img(indexQuery, :);
    T_te = txt(indexQuery, :);
    L_te = LAll(indexQuery, :);

    %% retrieval set
    index(indQ) = [];
    indexRetrieval = index;
    I_db = img(indexRetrieval, :);
    T_db = txt(indexRetrieval, :);
    L_db = LAll(indexRetrieval, :);

    %% training set
    N = 5000; param.N = N; %N is the number of traning data
    alpha = 0; %partial data ratio
    beta = 0.5; %image/tag ratio
    nc = N - alpha * N; param.nc = nc; %nc is the number of the image-tag pairs

    %image-tag pairs
    itrain = randperm(length(I_db), nc);
    I_tr = I_db(itrain, :);
    T_tr = T_db(itrain, :);
    L_tr = L_db(itrain, :);

    % unpaired image points
    n1 = round(beta * (alpha * N)); %n1 is the number of unpaired image points
    param.n1 = n1;
    itrain = randperm(length(I_db), param.n1);
    I1 = I_db(itrain, :);
    L1 = L_db(itrain, :);

    % unpaired tag points
    n2 = round((1 - beta) * (alpha * N)); %n2 is the number of unpaired tag points
    param.n2 = n2;
    itrain = randperm(length(I_db), param.n2);
    T2 = T_db(itrain, :);
    L2 = L_db(itrain, :);
    fprintf('the number of traning data = %d\n', N);
    fprintf('\n');
    fprintf('alpha = %0.1f%%\n', alpha);
    fprintf('beta = %0.1f%%\n', beta);
    fprintf('\n');
    fprintf('the number of the image-tag pairs is %d\n', nc);
    fprintf('the number of the partial data is %d (unpaired image samples is %d + unpaired tag samples is %d)\n', N - nc, n1, n2);
