function [P] = evaluateMFKH(data, param)
    %% train
    tic
    param = trainMFKH(data, param);
    toc
    %% hashing
    tic
    tstdata = data.test_kdata;
    dbdata = data.db_kdata;
    groundtruth = data.groundtruth;
    [B_db, U] = compressMFKH(dbdata, param);
    [B_tst, U] = compressMFKH(tstdata, param);
    %% evaluate
    Dhamm = hammingDist(B_db + 2, B_tst + 2);
    [P] = perf_metric4Label(data.db_labels', data.test_labels', Dhamm);
    toc
