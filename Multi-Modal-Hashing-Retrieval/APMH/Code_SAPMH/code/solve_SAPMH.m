function [W, U1, U2, param] = solve_SAPMH(I_tr, T_tr, I1, T2, Sc, S1, S2, S, param)
    %% random initialization
    k = param.k;
    theta = param.theta;
    lambda = param.lambda;
    rho = param.rho;
    n1 = param.n1;
    n2 = param.n2;
    n_anchors = param.n_anchors;
    ntrain = param.ntrain;
    bits = param.bits;
    param.k = k;
    mu1 = 0.5;
    mu2 = 0.5;

    sample = randsample(ntrain, n_anchors);
    anchorI = I_tr(sample, :);
    anchorT = T_tr(sample, :);
    % sigmaI=100;
    % sigmaT=100;% MIR Flickr

    sigmaI = 88;
    sigmaT = 88; % NUS-WIDE
    param.anchorI = anchorI;
    param.anchorT = anchorT;
    param.sigmaI = sigmaI;
    param.sigmaT = sigmaT;

    PhiI = exp(-sqdist(I_tr, anchorI) / (2 * sigmaI * sigmaI));
    PhtT = exp(-sqdist(T_tr, anchorT) / (2 * sigmaT * sigmaT));
    PhiI1 = exp(-sqdist(I1, anchorI) / (2 * sigmaI * sigmaI));
    PhtT2 = exp(-sqdist(T2, anchorT) / (2 * sigmaT * sigmaT));

    %initialize B
    Bc = randn(ntrain, bits) > 0; Bc = Bc * 2 - 1;
    B1 = randn(n1, bits) > 0; B1 = B1 * 2 - 1;
    B2 = randn(n2, bits) > 0; B2 = B2 * 2 - 1;
    B = [Bc; B1; B2];

    %initialize H
    Hc = randn(ntrain, k);
    H1 = randn(n1, k);
    H2 = randn(n2, k);
    H = [Hc; H1; H2];

    %initialize U
    U1 = (mu1 * Hc' * Hc + lambda * eye(k)) \ (mu1 * Hc' * PhiI);
    U2 = (mu2 * Hc' * Hc + lambda * eye(k)) \ (mu2 * Hc' * PhtT);

    %initialize W
    W = (H' * H) \ (theta * H' * B + rho * H' * S' * B) / (theta * eye(bits) + mu2 * B' * B);

    %initialize  Z and G
    Z = randn(ntrain + n1 + n2, bits) > 0; Z = Z * 2 - 1;
    G = B - Z;

    threshold = 0.01;
    lastF = 10000;
    iter = 1;
    %% Iterative algorithm

    while (iter < 100)
        %update H
        Hc = (mu1 * PhiI * U1' + mu2 * PhtT * U2' + theta * Bc * W' + rho * Sc' * Bc * W') / (mu1 * (U1 * U1') + mu2 * (U2 * U2') + theta * (W * W') + rho * W * Bc' * Bc * W');
        H1 = (mu1 * PhiI1 * U1' + theta * B1 * W' + rho * S1' * B1 * W') / (mu1 * (U1 * U1') + theta * (W * W') + rho * W * B1' * B1 * W');
        H2 = (mu2 * PhtT2 * U2' + theta * B2 * W' + rho * S2' * B2 * W') / (mu2 * (U2 * U2') + theta * (W * W') + rho * W * B2' * B2 * W');
        H = [Hc; H1; H2];

        %update U
        U1 = (mu1 * (Hc' * Hc) + mu1 * H1' * H1 + lambda * eye(k)) \ (mu1 * Hc' * PhiI + mu1 * H1' * PhiI1);
        U2 = (mu2 * (Hc' * Hc) + mu1 * H2' * H2 + lambda * eye(k)) \ (mu2 * Hc' * PhtT + mu1 * H2' * PhtT2);

        %update W
        W = (H' * H) \ (theta * H' * B + rho * H' * S' * B) / (theta * eye(bits) + mu2 * B' * B);

        %update B
        B = sign(2 * theta * H * W + 2 * rho * bits * S * H * W - rho * Z * W' * (H' * H) * W + 0.01 * Z - G);
        Bc = B(1:ntrain, :);
        B1 = B(ntrain + 1:ntrain + n1, :);
        B2 = B(ntrain + n1 + 1:end, :);

        %update Z, G and eta
        Z = sign(-rho * B * W' * (H' * H) * W + 0.1 * B + G);
        G = G + 0.1 * (B - Z);

        %update mu
        H11 = [Hc; H1];
        H22 = [Hc; H2];
        X11 = [PhiI; PhiI1];
        X22 = [PhtT; PhtT2];
        mu1 = 1 / (2 * sqrt(sum(sum((X11 - H11 * U1).^2))));
        mu2 = 1 / (2 * sqrt(sum(sum((X22 - H22 * U2).^2))));

        norm1 = sum(sum((X11 - H11 * U1).^2));
        norm2 = sum(sum((X22 - H22 * U2).^2));
        norm3 = sum(sum((B - H * W).^2));
        norm4 = sum(sum((S - B * (H * W)').^2));
        norm5 = sum(sum((U1).^2)) + sum(sum((U2).^2));
        currentF = mu1 * norm1 + mu2 * norm2 + theta * norm3 + rho * norm4 +lambda * norm5;
        fprintf('currentF at iteration %d: %.2f; obj: %.4f\n', iter, currentF, lastF - currentF);

        if (lastF - currentF) < threshold

            if iter > 3
                break
            end

        end

        iter = iter + 1;
        lastF = currentF;
    end

end
