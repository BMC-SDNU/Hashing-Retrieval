%% please download CIFAR multiple feature data from my homepage
%% http://www.nlsde.buaa.edu.cn/~xlliu

% run demo.m to see how the codes work.%
% For any problem with the codes, feel free to contact me via xlliu@nlsde.buaa.edu.cn. Also, I hope you
% can cite our ACM MM'12 paper:
% @InProceedings{MM:MFKH,
%    Author = {Liu, Xianglong and He, Junfeng and Liu, Di and Lang, Bo},
%    Title = {Compact Kernel Hashing with Multiple Features},
%    BookTitle = {ACM International Conference on Multimedia (MM)},
%    Year = {2012}
% }
%
% Xianglong Liu
% 07/25/2013
function [] = demo_MFKH(bits, dataname)
    %myFun - Description
    %
    % Syntax: [] = demo_MFKH(bits, dataname)
    %
    % Long description

    addpath('tools');
    addpath('MFKH');
    addpath('LSH');

    ko = 7;
    nlandmarks = 300;
    sigmas = [0.04 0.04]; %parameters for gauss kernel, need be tuned for different data;
    nbits = [8];
    pos = [10 50 100 300 500 1000 2000 3000];
    run = 5;

    for rrr = 1:run
        tic
        exp_data = constructDataset(ko, nlandmarks, sigmas, dataname); %using the released CIFAR data
        toc

        %% Multiple Feature Kernel Hashing
        MFKHparam.max_iter = 10;
        MFKHparam.lambda = 1; %0.1
        MFKHparam.nbits = str2num(bits);
        MFKHparam.tol = 1e-5;
        MFKHparam.pos = pos;
        [P] = evaluateMFKH(exp_data, MFKHparam);
        map(rrr) = P;

    end

    fprintf('[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));

end
