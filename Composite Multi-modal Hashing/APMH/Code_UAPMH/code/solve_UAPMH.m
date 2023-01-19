function [W, U1, U2, param] = solve_UAPMH(I_tr, T_tr, I1, T2, param)
    % Initialization
    mu1 = 0.5;
    mu2 = 0.5;
    n_anchors = param.n_anchors;
    nc = param.nc;
    bit = param.bit;
    k = param.k;
    theta = param.theta;
    lambda = param.lambda;
    n1 = param.n1;
    n2 = param.n2;

    % kerne feature mapping
    sample = randsample(nc, n_anchors);
    anchorI = I_tr(sample, :);
    anchorT = T_tr(sample, :);
    sigmaI = 100;
    sigmaT = 100;
    param.anchorI = anchorI;
    param.anchorT = anchorT;
    param.sigmaI = sigmaI;
    param.sigmaT = sigmaT;
    PhiI = exp(-sqdist(I_tr, anchorI) / (2 * sigmaI * sigmaI));
    PhtT = exp(-sqdist(T_tr, anchorT) / (2 * sigmaT * sigmaT));
    PhiI1 = exp(-sqdist(I1, anchorI) / (2 * sigmaI * sigmaI));
    PhtT2 = exp(-sqdist(T2, anchorT) / (2 * sigmaT * sigmaT));

    %initialize B
    Bc = randn(nc, bit) > 0; Bc = Bc * 2 - 1;
    B1 = randn(n1, bit) > 0; B1 = B1 * 2 - 1;
    B2 = randn(n2, bit) > 0; B2 = B2 * 2 - 1;
    B = [Bc; B1; B2];

    %initialize H
    Hc = randn(nc, k);
    H1 = randn(n1, k);
    H2 = randn(n2, k);
    H = [Hc; H1; H2];

    %initialize U
    U1 = (mu1 * Hc' * Hc + lambda * eye(k)) \ (mu1 * Hc' * PhiI);
    U2 = (mu2 * Hc' * Hc + lambda * eye(k)) \ (mu2 * Hc' * PhtT);

    %initialize W
    W = (theta * H' * H + lambda * eye(k)) \ (theta * H' * B);

    threshold = 0.001;
    lastF = 100;
    iter = 1;
    %% Iterative algorithm
    while (iter < 100)
        %update H
        Hc = (mu1 * PhiI * U1' + mu2 * PhtT * U2' + theta * Bc * W') / (mu1 * (U1 * U1') + mu2 * (U2 * U2') + theta * (W * W'));
        H1 = (mu1 * PhiI1 * U1' + theta * B1 * W') / (mu1 * (U1 * U1') + theta * (W * W'));
        H2 = (mu2 * PhtT2 * U2' + theta * B2 * W') / (mu2 * (U2 * U2') + theta * (W * W'));
        H = [Hc; H1; H2];

        %update U
        U1 = (mu1 * (Hc' * Hc) + lambda * eye(k)) \ (mu1 * Hc' * PhiI);
        U2 = (mu2 * (Hc' * Hc) + lambda * eye(k)) \ (mu2 * Hc' * PhtT);

        %update W
        W = (theta * H' * H + lambda * eye(k)) \ (theta * H' * B);

        %update B
        B = sign(H * W);
        Bc = B(1:nc, :);
        B1 = B(nc + 1:nc + n1, :);
        B2 = B(nc + n1 + 1:end, :);

        %update mu
        H11 = [Hc; H1];
        H22 = [Hc; H2];
        X11 = [PhiI; PhiI1];
        X22 = [PhtT; PhtT2];
        mu1 = 1 / (2 * sqrt(sum(sum((X11 - H11 * U1).^2))));
        mu2 = 1 / (2 * sqrt(sum(sum((X22 - H22 * U2).^2))));
        %     mu1 =  1 ;
        %     mu2  = 1 ;
        norm1 = sum(sum((X11 - H11 * U1).^2));
        norm2 = sum(sum((X22 - H22 * U2).^2));
        norm3 = sum(sum((B - H * W).^2));
        norm4 = sum(sum((U1).^2)) + sum(sum((U2).^2)) + sum(sum((W).^2));
        currentF = mu1 * norm1 + mu2 * norm2 + theta * norm3 + lambda * norm4;
        fprintf('currentF at iteration %d: %.2f; obj: %.4f\n', iter, currentF, lastF - currentF); %��ʾĿ�꺯��ֵ

        if (lastF - currentF) < threshold

            if iter > 3
                break
            end

        end

        iter = iter + 1;
        lastF = currentF;
    end

    param.mu1 = mu1;
    param.mu2 = mu2;
end
