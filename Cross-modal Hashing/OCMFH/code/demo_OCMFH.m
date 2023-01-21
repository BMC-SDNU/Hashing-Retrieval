function [] = demo_OCMFH(bits, dataname)
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
    
    numbatch = 2000;             
    [streamdata,nstream,L_tr,I_tr,T_tr] = predata_stream(I_tr,T_tr,L_tr,numbatch);
    %% Calculate the groundtruth
    GT = L_te*L_tr';
    WtrueTestTraining = zeros(size(L_te,1),size(L_tr,1));
    WtrueTestTraining(GT>0)=1;
    %% Parameter setting
    bit = bits; 
    %% Learn OCMFH
    [B_I,B_T,tB_I,tB_T,PI,PT, HH] = main_OCMFH(streamdata, I_te, T_te, bit);
    
    %% database hashing code
    Yi_db = sign((bsxfun(@minus,PI * I_db' , mean(HH,2)))');
    Yt_db = sign((bsxfun(@minus,PT * T_db' , mean(HH,2)))');
    Yi_db(Yi_db<0) = 0;
    Yt_db(Yt_db<0) = 0;
    B_I_db = compactbit(Yi_db);
    B_T_db = compactbit(Yt_db);
    %% Compute mAP
    Dhamm = hammingDist(tB_I, B_I_db)';    %image->text
    [~, HammingRank]=sort(Dhamm,1);
    mapIT = map_rank(L_db,L_te,HammingRank); 
    Dhamm = hammingDist(tB_T, B_T_db)';    %text->image
    [~, HammingRank]=sort(Dhamm,1);
    mapTI = map_rank(L_db,L_te,HammingRank); 
    map = [mapIT(100),mapTI(100)];
    
    MAP_I2T = map(1)
    MAP_T2I = map(2)

    fprintf('----------------result------------------\n')
    fprintf('map(I->T) = %.4f  map(T->I) = %.4f\n', map(1), map(2));

    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, bits, MAP_I2T, MAP_T2I);

end