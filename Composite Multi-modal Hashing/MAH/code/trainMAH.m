function MAHparam = trainMAH(data, param)
%% number of feature types

train_data = data.train_kdata;

%% parameter setting
M = 2;
alpha = ones(M, 1)/M;
nbits = param.nbits;
gamma = param.gamma;
eta = param.eta;
%GNMF learning
options = [];
options.WeightMode = 'Binary';  
% W = constructW(fea,options);
W1 = constructW(train_data.instance{1}, options);
W2 = constructW(train_data.instance{2}, options);
options.error = 0;
options.nRepeat = 1;
options.maxIter = 100;
options.minIter = 1;
options.meanFitRatio = 1;
options.alpha = alpha;
options.nbits = nbits;
options.gamma = gamma;
options.eta = eta;
% nClass = length(unique(gnd));
nClass = options.nbits;
[U,V,alpha] = MAH(train_data,nClass,W1,W2,options);

%% hash function generation
V_thres = sign(V);
[optTheta,functionVal,exitFlag] = Gradient_descent(V, V_thres);

MAHparam.alpha = alpha;
MAHparam.U = U;
MAHparam.theta = optTheta;