function [W, mu1, mu2, output, wxz] = solveFSMH(phi_x1, phi_x2, bits, param, S)
    %% random initializationcq
    [col, row] = size(phi_x1); 
    [~, rowt] = size(phi_x2);
    n_anchors = param.n_anchors;
    alpha = param.alpha;
    beta = param.beta;
    theta = param.theta;
    eta = param.eta;
    t = param.t;
    rhoo = param.rhoo;
    mu1 = 0.6;
    mu2 = 0.4;
    
    B = randn(col, bits) > 0; B = B * 2 - 1;
    Z = randn(col, bits) > 0; Z = Z * 2 - 1;
    H = B - Z;
    Phi_X = [sqrt(mu1) * phi_x1, sqrt(mu2) * phi_x2];
    V = randn(col, bits);
    R = (alpha * B' * B + beta * eye(bits)) \ (alpha * B' * S * V + beta * B' * V) / (V' * V);
    U = (Phi_X' * V) / (V' * V);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    U1 = U(1:row, :) ./ sqrt(mu1);
    U2 = U(row + 1:end, :) ./ sqrt(mu2);
    W = (Phi_X' * Phi_X + theta * eye(2 * n_anchors + 2)) \ (Phi_X' * B);
    threshold = 0.0001;
    lastF = 10000000000;
    iter = 1;
    MaxIter = 20; % 100

    wxz.x_index = 1:MaxIter;
    wxz.norm1 = 1:MaxIter;
    wxz.norm2 = 1:MaxIter;
    wxz.norm3 = 1:MaxIter;
    wxz.F = 1:MaxIter;

    %% Iterative algorithm
    while (iter < MaxIter)
        Phi_X = [sqrt(mu1) * phi_x1, sqrt(mu2) * phi_x2];
        
        G = -2 * Phi_X' * V;
        [P, ~, Q] = svd(G, 'econ');
        U = P * Q';
        Z = sign(-alpha * B * (R * V' * V * R') +eta * B + H);
        H = H + eta * (B - Z);
        eta = rhoo * eta;

        U1 = U(1:row, :) ./ sqrt(mu1);
        U2 = U(row + 1:end, :) ./ sqrt(mu2);
        R = (alpha * B' * B + beta * eye(bits)) \ (alpha * B' * S * V + beta * B' * V) / (V' * V);
        V = (Phi_X * U + alpha * S' * B * R + beta * B * R) / (U' * U + alpha * R' * B' * B * R + beta * R' * R);

        B = sign(-alpha * Z * R * V' * V * R' + 2 * alpha * S * V * R' + 2 * beta * V * R' + eta * Z - H);
        h1 = sum(sum((phi_x1' - U1 * V').^2));
        h2 = sum(sum((phi_x2' - U2 * V').^2));
        mu1 = ((1 / h1).^(1 / (t - 1))) / ((1 / h1).^(1 / (t - 1)) + (1 / h2).^(1 / (t - 1)));
        mu2 = ((1 / h2).^(1 / (t - 1))) / ((1 / h1).^(1 / (t - 1)) + (1 / h2).^(1 / (t - 1)));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        norm1 = sum(sum((Phi_X' - U * V').^2));
        norm2 = sum(sum((S - B * R * V').^2));
        norm3 = sum(sum((B' - R * V').^2));
        currentF = norm1 + alpha * norm2 + beta * norm3;

        wxz.norm1(iter) = norm1;
        wxz.norm2(iter) = norm2;
        wxz.norm3(iter) = norm3;
        wxz.F(iter) = currentF;

        fprintf('currentF at iteration %d: %.2f; obj: %.4f\n', ...
            iter, currentF, lastF - currentF);

        if (lastF - currentF) < threshold

            if iter > 3
                break
            end

        end

        iter = iter + 1;
        lastF = currentF;
    end

    W = (Phi_X' * Phi_X + theta * eye(2 * n_anchors + 2)) \ (Phi_X' * B);
    output.xdim = size(Phi_X);
    output.anchor = n_anchors;
    output.Bdim = size(B);
end
