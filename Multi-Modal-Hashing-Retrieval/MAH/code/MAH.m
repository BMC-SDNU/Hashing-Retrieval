function [U_final, V_final, alpha_final] = MAH(train_data, k, W1, W2, options)
    % Graph regularized Non-negative Matrix Factorization (GNMF) with
    %          multiplicative update
    %
    % where
    %   X
    % Notation:
    % X ... (mFea x nSmp) data matrix
    %       mFea  ... number of words (vocabulary size)
    %       nSmp  ... number of documents
    % k ... number of hidden factors
    % W ... weight matrix of the affinity graph
    %
    % options ... Structure holding all settings
    %
    % You only need to provide the above four inputs.
    %
    % X = U*V'
    %
    % References:
    % [1] Deng Cai, Xiaofei He, Xiaoyun Wu, and Jiawei Han. "Non-negative
    % Matrix Factorization on Manifold", Proc. 2008 Int. Conf. on Data Mining
    % (ICDM'08), Pisa, Italy, Dec. 2008.
    %
    % [2] Deng Cai, Xiaofei He, Jiawei Han, Thomas Huang. "Graph Regularized
    % Non-negative Matrix Factorization for Data Representation", IEEE
    % Transactions on Pattern Analysis and Machine Intelligence, , Vol. 33, No.
    % 8, pp. 1548-1560, 2011.
    %
    %
    %   version 2.1 --Dec./2011
    %   version 2.0 --April/2009
    %   version 1.0 --April/2008
    %
    %   Written by Deng Cai (dengcai AT gmail.com)
    %

    differror = options.error;
    maxIter = options.maxIter;
    nRepeat = options.nRepeat;
    minIter = options.minIter - 1;

    if ~isempty(maxIter) && maxIter < minIter
        minIter = maxIter;
    end

    meanFitRatio = options.meanFitRatio;

    alpha = options.alpha;
    nbits = options.nbits;
    gamma = options.gamma;
    eta = options.eta;

    Norm = 2;
    NormV = 0;

    %% Initialize
    K = 0;

    if alpha > 0
        K1 = alpha(1) * train_data.instance{1};
        K2 = alpha(2) * train_data.instance{2};
        K = alpha(1) * train_data.instance{1} + alpha(2) * train_data.instance{2};
    end

    [mFea, nSmp] = size(K);

    if alpha > 0
        W1 = alpha(1) * W1;
        DCol = full(sum(W1, 2));
        D1 = spdiags(DCol, 0, nSmp, nSmp);
        L1 = D1 - W1;

        if isfield(options, 'NormW') && options.NormW
            D_mhalf = spdiags(DCol.^ - .5, 0, nSmp, nSmp);
            L1 = D_mhalf * L1 * D_mhalf;
        end

    else
        L1 = [];
    end

    if alpha > 0
        W2 = alpha(2) * W2;
        DCol = full(sum(W2, 2));
        D2 = spdiags(DCol, 0, nSmp, nSmp);
        L2 = D2 - W2;

        if isfield(options, 'NormW') && options.NormW
            D_mhalf = spdiags(DCol.^ - .5, 0, nSmp, nSmp);
            L2 = D_mhalf * L2 * D_mhalf;
        end

    else
        L2 = [];
    end

    W = alpha(1) * W1 + alpha(2) * W2;
    L = alpha(1) * L1 + alpha(2) * L2;
    D = alpha(1) * D1 + alpha(2) * D2;

    selectInit = 1;

    U = abs(rand(mFea, k));
    V = abs(rand(k, nSmp));
    % initialize U
    KV = K * V'; % mnk or pk (p<<mn)
    VV = V * V'; % mk^2
    UVV = U * VV; % nk^2

    if alpha > 0
        UUU = (U * U') * U;
        KV = KV + 2 * eta * U;
        UVV = UVV + 2 * eta * UUU;
    end

    U = U .* (KV ./ max(UVV, 1e-10));
    % initialize V
    UK = U' * K; % mnk or pk (p<<mn)
    UU = U' * U; % mk^2
    UUV = UU * V; % nk^2

    if alpha > 0
        VW = V * W;
        VD = V * D;
        UK = UK + gamma * VW;
        UUV = UUV + gamma * VD;
    end

    V = V .* (UK ./ max(UUV, 1e-10));

    %% training model
    threshold = 1;
    lastF = 100000000000;
    iter = 1;

    while (true)
        % ===================== update V ========================
        UK = U' * K; % mnk or pk (p<<mn)
        UU = U' * U; % mk^2
        UUV = UU * V; % nk^2

        if alpha > 0
            VW = V * W;
            VD = V * D;

            UK = UK + gamma * VW;
            UUV = UUV + gamma * VD;
        end

        V = V .* (UK ./ max(UUV, 1e-10));

        % ===================== update U ========================
        KV = K * V'; % mnk or pk (p<<mn)
        VV = V * V'; % mk^2
        UVV = U * VV; % nk^2

        if alpha > 0
            UUU = (U * U') * U;

            KV = KV + 2 * eta * U;
            UVV = UVV + 2 * eta * UUU;
        end

        U = U .* (KV ./ max(UVV, 1e-10));

        % ===================== update alpha ========================
        T1 = trace(U * V * K1) - gamma * trace(V * L1 * V') / 2;
        T2 = trace(U * V * K2) - gamma * trace(V * L2 * V') / 2;
        A = [trace(K1 * K1) - trace(K1 * K2) trace(K2 * K1) - trace(K2 * K2); 1 1];
        B = [T1 - T2; 1];
        alpha = A \ B;
        % ===================== update Kernel ========================
        if alpha > 0
            K1 = alpha(1) * train_data.instance{1};
            K2 = alpha(2) * train_data.instance{2};
            K = alpha(1) * train_data.instance{1} + alpha(2) * train_data.instance{2};
        end

        if alpha > 0
            W1 = alpha(1) * W1;
            DCol = full(sum(W1, 2));
            D1 = spdiags(DCol, 0, nSmp, nSmp);
            L1 = D1 - W1;

            if isfield(options, 'NormW') && options.NormW
                D_mhalf = spdiags(DCol.^ - .5, 0, nSmp, nSmp);
                L1 = D_mhalf * L1 * D_mhalf;
            end

        end

        if alpha > 0
            W2 = alpha(2) * W2;
            DCol = full(sum(W2, 2));
            D2 = spdiags(DCol, 0, nSmp, nSmp);
            L2 = D2 - W2;

            if isfield(options, 'NormW') && options.NormW
                D_mhalf = spdiags(DCol.^ - .5, 0, nSmp, nSmp);
                L2 = D_mhalf * L2 * D_mhalf;
            end

        end

        W = alpha(1) * W1 + alpha(2) * W2;
        L = alpha(1) * L1 + alpha(2) * L2;
        D = alpha(1) * D1 + alpha(2) * D2;

        % ===================== compute the objective function ========================
        norm1 = sum(sum((K - U * V).^2));
        norm2 = alpha(1) * trace(V * L1 * V') + alpha(2) * trace(V * L2 * V');
        norm3 = sum(sum((U' * U - eye(nbits)).^2));
        currentF = norm1 + gamma * norm2 + eta * norm3;
        fprintf('\ncurrentF at iteration %d: %.2f; obj: %.4f\n', iter, currentF, lastF - currentF);

        if ((lastF - currentF) < threshold) || iter >= 10

            if iter > 1
                return;
            end

        end

        U_final = U;
        V_final = V;
        alpha_final = alpha;
        iter = iter + 1;
        lastF = currentF;
    end

    %==========================================================================

    function [obj, dV] = CalculateObj(X, U, V, L, deltaVU, dVordU)
        MAXARRAY = 500 * 1024 * 1024/8; % 500M. You can modify this number based on your machine's computational power.

        if ~exist('deltaVU', 'var')
            deltaVU = 0;
        end

        if ~exist('dVordU', 'var')
            dVordU = 1;
        end

        dV = [];
        nSmp = size(X, 2);
        mn = numel(X);
        nBlock = ceil(mn / MAXARRAY);

        if mn < MAXARRAY
            dX = U * V' - X;
            obj_NMF = sum(sum(dX.^2));

            if deltaVU

                if dVordU
                    dV = dX' * U + L * V;
                else
                    dV = dX * V;
                end

            end

        else
            obj_NMF = 0;

            if deltaVU

                if dVordU
                    dV = zeros(size(V));
                else
                    dV = zeros(size(U));
                end

            end

            PatchSize = ceil(nSmp / nBlock);

            for i = 1:nBlock

                if i * PatchSize > nSmp
                    smpIdx = (i - 1) * PatchSize + 1:nSmp;
                else
                    smpIdx = (i - 1) * PatchSize + 1:i * PatchSize;
                end

                dX = U * V(smpIdx, :)' - X(:, smpIdx);
                obj_NMF = obj_NMF + sum(sum(dX.^2));

                if deltaVU

                    if dVordU
                        dV(smpIdx, :) = dX' * U;
                    else
                        dV = dU + dX * V(smpIdx, :);
                    end

                end

            end

            if deltaVU

                if dVordU
                    dV = dV + L * V;
                end

            end

        end

        if isempty(L)
            obj_Lap = 0;
        else
            obj_Lap = sum(sum((V' * L) .* V'));
        end

        obj = obj_NMF + obj_Lap;

        function [U, V] = NormalizeUV(U, V, NormV, Norm)
            K = size(U, 2);

            if Norm == 2

                if NormV
                    norms = max(1e-15, sqrt(sum(V.^2, 1)))';
                    V = V * spdiags(norms.^ - 1, 0, K, K);
                    U = U * spdiags(norms, 0, K, K);
                else
                    norms = max(1e-15, sqrt(sum(U.^2, 1)))';
                    U = U * spdiags(norms.^ - 1, 0, K, K);
                    V = V * spdiags(norms, 0, K, K);
                end

            else

                if NormV
                    norms = max(1e-15, sum(abs(V), 1))';
                    V = V * spdiags(norms.^ - 1, 0, K, K);
                    U = U * spdiags(norms, 0, K, K);
                else
                    norms = max(1e-15, sum(abs(U), 1))';
                    U = U * spdiags(norms.^ - 1, 0, K, K);
                    V = V * spdiags(norms, 0, K, K);
                end

            end
