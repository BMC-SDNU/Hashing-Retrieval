function [P] = evaluateMAH(data, param)
    %% train
    tic
    MAHparam = trainMAH(data, param);
    %% hashing
    alpha = MAHparam.alpha;
    U = MAHparam.U;
    theta = MAHparam.theta;
    tstdata = data.test_kdata;
    dbdata = data.db_kdata;
    K_tstdata = alpha(1) * tstdata.instance{1} + alpha(2) * tstdata.instance{2};
    K_dbdata = alpha(1) * dbdata.instance{1} + alpha(2) * dbdata.instance{2};
    P = (U' * U) \ U';
    new_tstdata = P * K_tstdata;
    new_dbdata = P * K_dbdata;
    [B_tst] = round(h_func(new_tstdata, theta));
    [B_db] = round(h_func(new_dbdata, theta));
    toc
    %% evaluate
    tic
    Dhamm = hammingDist(B_db', B_tst');
    [P] = perf_metric4Label(data.db_labels', data.test_labels', Dhamm); %this
    toc
