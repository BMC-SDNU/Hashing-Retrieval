function [] = demo_ASCSH(bits, dataname)
    data_dir = '../../data';

    if ~exist(data_dir, 'dir')
        error('No such dir(%s)', fullfile(pwd, data_dir))
    end

    if ~exist('../result', 'dir')
        mkdir('../result')
    end

    addpath(data_dir);
    bits = str2num(bits);
    
    if strcmp(dataname, 'flickr')
        load('mir_cnn.mat');

    elseif strcmp(dataname, 'nuswide')
        load('nus_cnn.mat');
    elseif strcmp(dataname, 'coco')
        load('coco_cnn.mat');
    else
        fprintf('ERROR dataname!');
    end
    
    addpath(genpath(fullfile('utils/')));
    seed = 0;
    rng('default');
    rng(seed);
    param.seed = seed;
%     dataname = 'flickr-25k';
    %% parameters setting
    % basie information
    param.dataname = dataname;
    param.method = 'DLFH'; 

    % method parameters
%     bits = [8];
    bits = bits;
    nb = numel(bits);

    param.bits = bits;
    param.maxIter = 25;
    param.gamma = 1e-6;
    param.lambda = 8; %At 64 bits, it is suggested that this parameter be set to 10-12

    if strcmp(dataname, 'flickr')
        %flickr
        param.lambda_c = 1e-3;
        param.alpha1 = 5e-2;
        param.alpha2 = param.alpha1;
        param.mu = 1e-4;
        param.eta = 0.005;
    else
        %nus
        param.lambda_c = 1e-3;
        param.alpha1 = 5e-3;
        param.alpha2 = param.alpha1;
        param.mu = 1e-4;
        param.eta = 1e-2;
        param.sc = 5000;
    end
    %nus
    % param.lambda_c = 1e-3;
    % param.alpha1 = 5e-3;
    % param.alpha2 = param.alpha1;
    % param.mu = 1e-4;
    % param.eta = 1e-2;
    % param.sc = 5000;

    %iapr
    % param.lambda_c = 1e-5;
    % param.alpha1 = 5e-3;
    % param.alpha2 = param.alpha1;
    % param.mu = 1e-6;
    % param.eta = 1e-4;
    
    %% load dataset
%     dataset = load_data(dataname);
%     n_anchors = 1500;
%     rbf2;
      inx = randperm(size(L_tr,1),size(L_tr,1)); % ´òÂÒ²Ù×÷
      dataset.XTest = I_te; % test image
      dataset.YTest = T_te; % test text
      dataset.XDatabase = I_tr(inx,:); % train image
      dataset.YDatabase = T_tr(inx,:); % train text
      dataset.testL = L_te; % test Label
      dataset.databaseL = L_tr(inx,:); % train Label
      
      dataset.I_db = I_db;
      dataset.T_db = T_db;
      dataset.L_db = L_db;
      % image:n¡Á1500 text:n¡Á1500
      n_anchors = 1500;
      rbf2;
      
%       [n, ~] = size(I_db);
      I_db_x = RBF_fast(I_db',anchor_image');
      I_db_y = RBF_fast(T_db',anchor_text');

    %% run algorithm
    for i = 1: nb
        fprintf('...method: %s\n', param.method);
        fprintf('...bit: %d\n', bits(i)); 
        param.bit = bits(i);
        param.num_samples = 2 * param.bit; %At 64 bits, it is suggested that this parameter be set to £¨2.5 * param.bit£©
        trainL = dataset.databaseL;
        [C, D1, D2, ~]= ASCSH(trainL, param, dataset);
        
        % retrieval
        tBX = sign(dataset.XTest * (C+D1)'); % test image hashing code
        tBY = sign(dataset.YTest * (C+D2)'); % test text hashing code
        
        dBX = sign(I_db_x * (C+D1)'); % database image hashing code
        dBY = sign(I_db_y * (C+D1)'); % database image hashing code
        sim_ti = dBY * tBX'; % image -> text
        sim_it = dBY * tBY'; % text -> image
        R = size(dBX,1);
        ImgToTxt = mAP(sim_ti, L_db, L_te, R);
        TxtToImg = mAP(sim_it, L_db, L_te, R);
        MAP_I2T = ImgToTxt;
        MAP_T2I = TxtToImg;
        fprintf('--------------------result------------------\n')
        fprintf( '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, bits(i), ImgToTxt, TxtToImg);
        
        name = ['../result/' dataname '.txt'];
        fid = fopen(name, 'a+');
        fprintf(fid, '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, bits(i), MAP_I2T, MAP_T2I);

        
    end
    
    
    
    

end