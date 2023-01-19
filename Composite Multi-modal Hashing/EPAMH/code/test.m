function map = test(X, bit, method)
    %
    % demo code for generating small code and evaluation
    % input X should be a n*d matrix, n is the number of images, d is dimension
    % ''method'' is the method used to generate small code
    % ''method'' can be 'ITQ', 'RR', 'LSH' and 'SKLSH'
    %

    % parameters
    averageNumberNeighbors = 50; % ground truth is 50 nearest neighbor
    num_test = 1000; % 1000 query test point, rest are database
    bit = bit; % bits used

    % split up into training and test set
    [ndata, D] = size(X);
    R = randperm(ndata);
    Xtest = X(R(1:num_test), :);
    R(1:num_test) = [];
    Xtraining = X(R, :);
    num_training = size(Xtraining, 1);
    clear X;

    % define ground-truth neighbors (this is only used for the evaluation):
    R = randperm(num_training);
    DtrueTraining = distMat(Xtraining(R(1:100), :), Xtraining); % sample 100 points to find a threshold
    Dball = sort(DtrueTraining, 2);
    clear DtrueTraining;
    Dball = mean(Dball(:, averageNumberNeighbors));
    % scale data so that the target distance is 1
    Xtraining = Xtraining / Dball;
    Xtest = Xtest / Dball;
    Dball = 1;
    % threshold to define ground truth
    DtrueTestTraining = distMat(Xtest, Xtraining);
    WtrueTestTraining = DtrueTestTraining < Dball;
    clear DtrueTestTraining

    % generate training ans test split and the data matrix
    XX = [Xtraining; Xtest];
    % center the data, VERY IMPORTANT
    sampleMean = mean(XX, 1);
    XX = (XX - repmat(sampleMean, size(XX, 1), 1));

    %several state of art methods
    switch (method)
            % ITQ method proposed in our CVPR11 paper
        case 'ITQ'
            % PCA
            [pc, l] = eigs(cov(XX(1:num_training, :)), bit);
            XX = XX * pc;
            % ITQ
            [Y, R] = ITQ(XX(1:num_training, :), 50);
            XX = XX * R;
            Y = zeros(size(XX));
            Y(XX >= 0) = 1;
            Y = compactbit(Y > 0);
    end

    % compute Hamming metric and compute recall precision
    B1 = Y(1:size(Xtraining, 1), :);
    B2 = Y(size(Xtraining, 1) + 1:end, :);
    Dhamm = hammingDist(B2, B1);
    map = mAP(Dhamm, L_tr, L_te);
