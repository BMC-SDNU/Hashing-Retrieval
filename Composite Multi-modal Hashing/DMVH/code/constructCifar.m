clear; clc;
load('mirflickr.mat');

X1 = [normc(train_data_gist) normc(test_data_gist)];
X2 = [normc(train_data_sift) normc(test_data_sift)];
all_labels = [train_labels ; test_labels]';

%Preprocessing
X1=normZeroMean(X1);
X1=normEqualVariance(X1);
X2=normZeroMean(X2);
X2=normEqualVariance(X2);

%Data Partition
num_dataset=size(X1,2);
num_test=1000;
num_train = 3000;

indexdb=1:num_dataset-num_test;
indexTest=num_dataset-num_test+1:num_dataset;
indexTrain=1:num_train;

I_db = X1(:, indexdb);
I_tr = X1(:, indexTrain);
I_te = X1(:, indexTest);
T_db = X2(:, indexdb);
T_tr = X2(:, indexTrain);
T_te = X2(:, indexTest);
L_db = all_labels(:, indexdb);
L_tr = all_labels(:, indexTrain);
L_te = all_labels(:, indexTest);