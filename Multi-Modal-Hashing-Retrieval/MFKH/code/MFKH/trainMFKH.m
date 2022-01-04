function MFKHparam = trainMFKH(data, MFKHparam)

    %number of feature types
    tic;

    train_data = data.train_kdata;
    S = data.S;
    N = size(S, 1);
    M = data.views;
    nbits = MFKHparam.nbits;
    %intilize mu
    mu = ones(M, 1) / M;

    iter = 0;
    obj = Inf;
    %fprintf('Begin alternative optimization\n');

    D = full(sum(S));
    LS = diag(D.^ - 0.5) * (diag(D) - S) * diag(D.^ - 0.5);
    %% initialize D with the linear combination of all D_i

    % D = full(diag(sum(S)));
    % LS = D-S;
    while (iter < MFKHparam.max_iter)
        %% Learn W and b
        K_LN = 0;
        K_LL = 0;

        for i = 1:M
            K_LN = K_LN + mu(i) * train_data.instance{i};
            K_LL = K_LL + mu(i) * train_data.ll{i};
        end

        k = mean(K_LN, 2);
        C = K_LN * LS * K_LN' + MFKHparam.lambda * K_LL;
        C = (C + C') / 2;
        G = K_LN * K_LN' / N - k * k';
        G = (G + G') / 2;

        [T V] = eig(G);
        sigma = diag(V);
        com_idx = sigma > eps;
        T = T(:, com_idx);
        sigma_part = sigma(com_idx);
        Gamma = diag(sigma_part.^(-0.5));

        C_B = Gamma' * T' * C * T * Gamma;
        C_B = (C_B + C_B') / 2;
        [W_B V] = eig(C_B);
        sigma = diag(V);
        [~, idx] = sort(sigma);

        W = T * Gamma * W_B(:, idx(1:nbits)); %min_idx);

        b = W' * k;

        %% Learn mu
        E = zeros(M);
        h = zeros(M, 1);

        for i = 1:M

            for j = 1:M
                E(i, j) = trace(W' * train_data.instance{i} * LS * train_data.instance{j}' * W);
            end

            h(i) = MFKHparam.lambda * trace(W' * train_data.ll{i} * W);
        end

        E = (E + E');
        %     E = E + E' - diag(diag(E));
        Aeq = ones(1, M);
        beq = 1;
        lb = ones(M, 1) * eps;
        ub = ones(M, 1) - lb;

        pmu = mu;
        opts = optimset('Algorithm', 'interior-point-convex');
        mu = quadprog(E, h, [], [], Aeq, beq, lb, ub, pmu, opts);

        iter = iter + 1;

        old_obj = obj;

        %     Y = W'*K_LN-repmat(b, [1, N]);
        %     Y = sign(Y);%Y>0
        obj = mu' * E * mu / 2 + h' * mu; %trace(Y*LS*Y');

        if (old_obj < obj)
            mu = pmu;
            break;
        end

        %fprintf('\tinter %d, mu ', iter);
        %fprintf(', obj %f\n', obj);
        %for ii =1:M
        %   fprintf('%f ', pmu(ii));
        %end
    end

    %fprintf('End alternative optimization\n');

    MFKHparam.w = W;
    MFKHparam.b = b;
    MFKHparam.mu = mu;
    MFKHparam.views = M;
    MFKHparam.iter = iter;
    MFKHparam.train_time = toc;
