function [] = demo_DOCH(bits, dataname)
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
    
    % nbits_set=[8 16 32 64 128];
    nbits_set = bits;

    %% load dataset

    set = 'MIRFlickr';



    %% initialization
    fprintf('initializing...\n')
    alphas = [0.45 0.35 0.25 0.15 0.1];
    param.datasets = set;

    param.iter = 3;
    param.num_anchor = 50;
    param.theta = 0.1;
    % param.chunk_size = 2000;
    param.chunk_size = size(L_te,1); % 查询集长度
    
    % queryInds = R(1:2000);
    % sampleInds = R(2001:end);
    % param.nchunks = floor(length(sampleInds)/param.chunk_size);
    sampleInds = randperm(size(L_tr,1));
    param.nchunks = floor(size(L_tr,1)/param.chunk_size);
    fprintf('%d  \n',param.nchunks)
    XChunk = cell(param.nchunks,1);
    YChunk = cell(param.nchunks,1);
    LChunk = cell(param.nchunks,1);
    for subi = 1:param.nchunks-1
        XChunk{subi,1} = I_tr(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
        YChunk{subi,1} = T_tr(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
        LChunk{subi,1} = L_tr(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
    end
    XChunk{param.nchunks,1} = I_tr(sampleInds(param.chunk_size*subi+1:end),:);
    YChunk{param.nchunks,1} = T_tr(sampleInds(param.chunk_size*subi+1:end),:);
    LChunk{param.nchunks,1} = L_tr(sampleInds(param.chunk_size*subi+1:end),:);
    
    % XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
    XTest = I_te; YTest = T_te; LTest = L_te;

    


    for bit=1:length(nbits_set) 
        nbits = nbits_set(bit);
        param.alpha = alphas(bit);

       %% DOCH
        param.nbits=nbits;
        [Wx, Wy] = evaluate(XChunk,YChunk,LChunk,XTest,YTest,LTest,param);
        
         BxDatabase = compactbit(I_db*Wx >= 0); % image retrieval database
         ByDatabase = compactbit(T_db*Wy >= 0); %  retrieval database
        
       %% iamge as query to retrieve text database
        BxTest = compactbit(XTest*Wx >= 0);
       
        % ByTrain = compactbit(B >= 0);
        DHamm = hammingDist(BxTest, ByDatabase); % image->text
        [~, orderH] = sort(DHamm, 2);
        evaluation_info.Image_VS_Text_MAP  = mAP(orderH', L_db, LTest);
        MAP_I2T = evaluation_info.Image_VS_Text_MAP;
        
        %% text as query to retrieve image database
        ByTest = compactbit(YTest*Wy >= 0);
        % BxTrain = compactbit(B >= 0);
        DHamm = hammingDist(ByTest, BxDatabase); % text->image
        [~, orderH] = sort(DHamm, 2);
        evaluation_info.Text_VS_Image_MAP = mAP(orderH', L_db, LTest);
        MAP_T2I = evaluation_info.Text_VS_Image_MAP;
        fprintf('---------result----------\n')
        fprintf('DOCH %d bits ,   Image_VS_Text_MAP: %f,   Text_VS_Image_MAP: %f \n',param.nbits , evaluation_info.Image_VS_Text_MAP, evaluation_info.Text_VS_Image_MAP);
        name = ['../result/' dataname '.txt'];
        fid = fopen(name, 'a+');
        fprintf(fid, '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, param.nbits, MAP_I2T, MAP_T2I);
        fclose(fid);


    end


end