function exp_data = constructDataset(ko, nlandmarks, sigma, dataname)
    %data constrcution
    addpath('tools');
    
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

    rand('state', sum(100 * clock));
    test_data_gist = I_te';
    test_data_sift = T_te';
    test_labels = L_te;
    train_data_gist = I_db';
    train_data_sift = T_db';
    train_labels = L_db;
    train_num = 5000; %number of traning data
    test_num = 2243; %number of test data
    label_num = 24;
    nneighbors = ko; %15; %number of nneighbors of same labels
    kernels = {'gauss', 'gauss'}; %define kernel types
    sigmas = [sigma(1) sigma(2)];
    all_nviews = 2; %color, texture, sift hist
    views = [1 2]; %candidate: [1 2]
    nviews = length(views);

    data = [normc(train_data_sift) normc(test_data_sift); normc(train_data_gist) normc(test_data_gist)];
    [dim, data_num] = size(data);
    all_labels = [train_labels; test_labels]';
    L = sparse(all_labels);

    sift_sidx = 1;
    sift_dim = 1386;
    gist_sidx = 1387;
    gist_dim = 4096;
    view_idx = [sift_sidx sift_sidx + sift_dim - 1; gist_sidx gist_sidx + gist_dim - 1];

    db_labels = L;

    label_idx = find(sum(db_labels) > 0); %single or more labels
    perm = randperm(length(label_idx));
    test_idx = label_idx(perm(1:test_num));
    test_labels = db_labels(:, test_idx);

    for j = 1:all_nviews
        all_test_data.instance{j} = data(view_idx(j, 1):view_idx(j, 2), test_idx);
    end

    test_cdata = normc(data(view_idx(1, 1):view_idx(2, 2), test_idx));

    db_idx = 1:data_num;
    db_idx(test_idx) = [];
    data = data(:, db_idx);
    db_labels = db_labels(:, db_idx);
    db_cdata = normc(data(view_idx(1, 1):view_idx(2, 2), :));

    %% tain data
    label_idx = find(sum(db_labels) > 0);
    perm = randperm(length(label_idx));
    train_idx = label_idx(perm(1:train_num));
    train_labels = db_labels(:, train_idx);

    meanvec = mean(db_cdata(:, train_idx), 2);
    test_cdata = test_cdata - repmat(meanvec, 1, test_num);
    db_cdata = db_cdata - repmat(meanvec, 1, data_num - test_num);
    train_cdata = db_cdata(:, train_idx);

    S = zeros(train_num);

    for i = 1:train_num
        sim = train_labels(:, i)' * train_labels;
        sim(i) = 0;
        idx = find(sim > 0);

        Dist = distMat(train_cdata(:, i)', train_cdata(:, idx)');
        [~, dist_idx] = sort(Dist, 2);
        p_rand_idx = idx(dist_idx(1:nneighbors));
        S(i, p_rand_idx) = 1;
    end

    S = (S + S') > 0;

    for j = 1:all_nviews
        all_db_data.instance{j} = data(view_idx(j, 1):view_idx(j, 2), :);
        all_train_data.instance{j} = all_db_data.instance{j}(:, train_idx);
        all_Neighbors{j} = zeros(train_num, nneighbors);
        all_Distances{j} = zeros(train_num, nneighbors);

        all_S{j} = zeros(train_num);
        D = distMat(all_train_data.instance{j}', all_train_data.instance{j}');

        for i = 1:train_num
            Dist = D(i, :); %distMat(all_train_data.instance{j}',all_train_data.instance{j}');
            Dist(i) = Inf;
            [v, dist_idx] = sort(Dist, 2);
            rand_idx = dist_idx(1:nneighbors);

            all_S{j}(i, rand_idx) = v(1:nneighbors);

            all_Neighbors{j}(i, :) = rand_idx;
            all_Distances{j}(i, :) = v(1:nneighbors);
        end

        all_S{j} = (all_S{j} + all_S{j}');
        all_S{j} = sparse(all_S{j});
        dist = max(all_S{j}(:));
        all_S{j} = sparse((all_S{j} > 0) .* exp(- (all_S{j}) ./ dist));
    end

    for i = 1:test_num
        idx = test_labels(:, i) > 0;
        groundtruth{i} = find(test_labels(idx, i)' * db_labels(idx, :) > 0);
    end

    % if(exist('landmarks_cifar10_ind.mat', 'file'))
    %     load('landmarks_cifar10_ind.mat');
    % else
    for j = 1:all_nviews
        [C{j}, dis, label] = yael_kmeans (single(all_db_data.instance{j}), nlandmarks, 'niter', 50, 'verbose', 0);
    end

    %     save('landmarks_mirflickr_ind.mat', 'C', '-v7.3');
    % end
    for j = 1:nviews
        view = views(j);
        landmarks.instance{j} = C{view}; %(view_idx(view,1):view_idx(view,2), :);
        test_data.instance{j} = all_test_data.instance{view};
        test_kdata.instance{j} = KernelFunctions(kernels{view}, test_data.instance{j}, landmarks.instance{j}, sigmas(view));

        db_data.instance{j} = all_db_data.instance{view};
        db_kdata.instance{j} = KernelFunctions(kernels{view}, db_data.instance{j}, landmarks.instance{j}, sigmas(view));

        train_data.instance{j} = all_train_data.instance{view};
        train_kdata.instance{j} = db_kdata.instance{j}(:, train_idx);

        train_kdata.ll{j} = KernelFunctions(kernels{view}, landmarks.instance{j}, landmarks.instance{j}, sigmas(view));

        Neighbors{j} = all_Neighbors{view};
        Distances{j} = all_Distances{view};
    end

    exp_data.label_num = label_num;
    exp_data.nneighbors = nneighbors; %number of nneighbors per class
    exp_data.train_num = train_num; %length(trn_coarse_labels);
    exp_data.train_data = train_data;
    exp_data.train_kdata = train_kdata;
    exp_data.train_labels = full(train_labels);

    exp_data.test_data = test_data; %single label test
    exp_data.test_kdata = test_kdata;
    exp_data.test_labels = full(test_labels);
    exp_data.groundtruth = groundtruth;

    exp_data.S = sparse(S);
    exp_data.all_S = all_S;
    exp_data.db_data = db_data;
    exp_data.db_kdata = db_kdata;
    exp_data.db_labels = full(db_labels);

    exp_data.views = nviews;

    exp_data.Neighbors = Neighbors;
    exp_data.Distances = Distances;

    exp_data.db_cdata = db_cdata;
    exp_data.train_cdata = train_cdata;
    exp_data.test_cdata = test_cdata;

    exp_data.nlandmarks = nlandmarks;

    exp_data.all_test_data = all_test_data;
    exp_data.all_db_data = all_db_data;
    exp_data.all_train_data = all_train_data;
