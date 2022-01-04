function model = multiviewHashing(views, gamma, alpha, beta, L1, L2, nbits)

    % views contains all the training data from different views.
    % view1 and view2 should be m1*n, m2*n, labels should be n*c, where c is
    % the number binary code
    %
    % para is used for Laplacian Matrix learning
    % para.k
    % para.lamda
    %
    % gamma = 0.9;
    % alpha = 0.99;
    % beta = 0.9;

    gamma = 10^gamma;
    alpha = 10^alpha;
    beta = 10^beta;
    viewNum = size(views, 2);

    X = [views{1}'; views{2}'];
    [feaDimX, trainNum] = size(X);

    oneline = ones(trainNum, 1);
    eyemat = eye(trainNum);
    eyemat_m = eye(feaDimX);
    % eyematm = eye(trainF);
    Lc = eyemat - (oneline * oneline') / trainNum;

    % % Get Laplacian Matrix L1 and L2
    para.lamda = 1;
    para.k = 5;
    % [L1] = Laplacian_GK(views{1}, para);
    % [L2] = Laplacian_GK(views{2}, para);
    % save Laplacian_GK L1 L2;
    % load Laplacian_GK;
    % % ----------------------------------------------------------------
    % get w1 and L1
    temp = zeros(trainNum, trainNum);

    % B = Lc-Lc*X'/(X*Lc*X'+beta*eyemat_m)*X*Lc;
    % C1 = gamma*(I/(L1+gamma*eyemat));
    % C2 = gamma*(I/(L2+gamma*eyemat));

    D = gamma * eyemat - gamma * gamma * eyemat / (L1 + gamma * eyemat) + gamma * eyemat - gamma * gamma * eyemat / (L2 + gamma * eyemat) + alpha * (Lc - Lc * X' / (X * Lc * X' + beta * eyemat_m) * X * Lc);

    % clear L1 L2;
    D = (D + D') / 2;
    [v, eigval] = eig(D);
    eigval = diag(eigval);
    [eigval, idx] = sort(diag(eigval));
    topk = nbits;
    Y = v(idx(1:topk), :);
    Y = Y';
    % save Y Y;
    W = (X * Lc * X' + beta * eyemat_m) \ X * Lc * Y;
    % save W W;
    b = (oneline' * Y - oneline' * X' * W) / trainNum;
    % save b b;

    model.v = v;
    model.eigval = eigval;
    model.type = 'mfh';
    model.nbits = nbits;
    model.alpha = alpha;
    model.beta = beta;
    model.gamma = gamma;
    model.W = W;
    model.b = b;
    model.Y = Y;
