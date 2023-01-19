function evaluation_info=evaluateLSH(data, param)

dbdata = data.db_data;
tstdata = data.test_data;
trndata = data.train_data;
groundtruth = data.groundtruth;
clear data;

param = trainLSH(double(trndata'),param);

%%% Compression Time
[B_db, ~] = compressLSH(double(dbdata'), param);
[B_tst, ~] = compressLSH(double(tstdata'), param);

pos = param.pos;
poslen = length(pos);
label_r = zeros(1, poslen);
label_p = zeros(1, poslen);
for n = 1:size(tstdata, 2)
    % compute your distance
    D_code = hammingDist(B_tst(n,:),B_db);
    D_truth = groundtruth{n};%ground truth
     
    [P, R] = precall(D_code, D_truth, pos);

    label_r = label_r + R(1:poslen);
    label_p = label_p + P(1:poslen);
end
evaluation_info.recall=label_r/size(tstdata, 2);
evaluation_info.precision=label_p/size(tstdata, 2);
evaluation_info.LSHparam=param;