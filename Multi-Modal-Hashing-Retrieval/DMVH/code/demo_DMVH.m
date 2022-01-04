function [] = demo_DMVH(bits, dataname)
    %myFun - Description
    %
    % Syntax: [] = demo_DMVH(bits, dataname)
    %
    % Long description

    warning off;
    addpath('tools');
    addpath('LSH');
    %% parameter setting
    ko = 7;
    nlandmarks = 300;
    sigmas = [64 38]; %parameters for gauss kernel, need be tuned for different data;
    pos = [10 50 100 300 500 1000 2000 3000];
    %%
    nround = 3;
    map = zeros(nround, 1);

    for n = 1:nround
        tic
        exp_data = constructDataset(ko, nlandmarks, sigmas, dataname); %using the released CIFAR data
        toc

        %% Multiple Feature Kernel Hashing
        fprintf('algorithm\n');
        param.max_iter = 10;
        param.beta = 0.1; %0.01
        param.gamma = 0.01; %0.001
        param.nbits = str2num(bits);
        param.tol = 1e-5;
        param.pos = pos;
        param.M = 2;
        [P] = evaluateDMVH(exp_data, param);
        map(n, 1) = P;
    end

    fprintf('[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
end
