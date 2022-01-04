function [map] = evaluateDMVH(data, param)
    %% train
    tic
    DMVHparam = trainDMVH(data, param);
    toc
    %% hashing
    tic
    tstdata = data.test_kdata;
    dbdata = data.db_kdata;
    groundtruth = data.groundtruth;
    tic;
    [B_db, Udb] = compressDMVH(dbdata, DMVHparam);
    [B_tst, Utst] = compressDMVH(tstdata, DMVHparam);
    param.compress_time = toc;
    %% evaluate
    Dhamm = hammingDist(B_db, B_tst);
    [map] = perf_metric4Label(data.db_labels', data.test_labels', Dhamm); %this
    toc
