function [I_te, T_te, L_te, I_db, T_db, L_db, I_tr, T_tr, L_tr] = construct_mir(~)
    load('D:\Hashing produce\datasets\mir_5.mat')
    img = [I_db; I_te];
    txt = [T_db; T_te];
    LAll = [L_db; L_te];

    N = size(LAll, 1);
    index = [1:N];
    index = index';
    %% query set
    for i = 1:24 
        [row{i}, ~] = find(LAll(:, i) == 1);
        n_size = size(row{i}, 1);
        R = randperm(n_size, 100); 
        indQ{i} = row{i}(R, :);
    end

    j = 1; n = 100;

    for i = 1:24
        ind(j:n, :) = indQ{1, i}(:, :);
        j = j + 100;
        n = n + 100;
    end

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
    ntrain = 5000; 
    itrain = randperm(length(I_db), ntrain); 
    I_tr = I_db(itrain, :);
    T_tr = T_db(itrain, :);
    L_tr = L_db(itrain, :);
