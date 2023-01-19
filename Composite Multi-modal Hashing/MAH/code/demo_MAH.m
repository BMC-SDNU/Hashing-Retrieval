function [] = demo_MAH(bits, dataname)

    warning off;
    addpath('tools');
    addpath('Tool');
    % addpath('LSH');
    addpath(genpath(pwd))
    %% parameter setting
    run = 5;

    for rrr = 1:run
        ko = 7;
        nlandmarks = 300;
        sigmas = [0.0001 0.0001]; %parameters for gauss kernel, need be tuned for different data;
        nbits = str2num(bits);
        pos = [10 50 100 300 500 1000 2000 3000];
        exp_data = constructDataset(ko, nlandmarks, sigmas, dataname); 

        %% Multiple Feature Kernel Hashing
        param.gamma = 10; %10
        param.eta = 1; %1
        param.nbits = nbits;
        param.tol = 1e-5;
        param.pos = pos;
        param.M = 2;

        [P] = evaluateMAH(exp_data, param);
        map(rrr) = P;
    end

    fprintf('[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
end
