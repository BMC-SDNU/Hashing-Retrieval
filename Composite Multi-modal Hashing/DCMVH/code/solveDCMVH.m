function [opt] = solveDCMVH(Xa, Xb, param)
    % Xa & Xb           the feature matrices of the 1st&2nd views
    % Wa1 & Wb1    the feature selection matrices of the 1st layer in two view-specific networks
    % Ua1 & Ub1    the output of the 1st layer in two view-specific networks
    % Wa2 & Wb2    the transformation matrices of the 2nd layer in two view-specific networks
    % Ua2 & Ub2    the output of the 2nd layer in two view-specific networks
    % Wa3 & Wb3    the transformation matrices of the 3rd layer in two view-specific networks
    % H            the consensus multi-modal factor
    % W4           the orthogonal transformation matrix of the 4th layer of the fusion network
    % B            the hash code matrix
    % Y            the instance-wise semantic label matrix
    % mu1 & mu2    the weights of the 1st&2nd views

    %% Setting parameters
    theta = param.theta;
    alpha = param.alpha;
    beta = param.beta;
    delta = param.delta;
    gamma = param.gamma;
    rho = param.rho;
    [row1, col] = size(Xa);
    [row2, ~] = size(Xb);
    bits = param.bits;
    Y = param.Y;
    [c, ~] = size(Y);
    d1 = round(row1 / 2);
    mu1 = 0.5;
    mu2 = 0.5;
    t = 3;
    lastF = 100000000;
    threshold = 0.01;
    iter = 1;
    maxIter = 30;

    %% Matrix initialization
    B = sign(-1 + (1 - (-1)) * rand(bits, col));
    Zb = sign(-1 + (1 - (-1)) * rand(bits, col));
    Gb = B - Zb;
    Wa1 = randn(d1, row1);
    Wb1 = randn(d1, row2);
    Wa2 = randn(c, d1);
    Wb2 = randn(c, d1);
    Wa3 = randn(bits, c);
    Wb3 = randn(bits, c);
    W4 = randn(bits, bits);
    [Pw, ~, Qw] = svd(W4, 'econ');
    W4 = Pw * Qw';
    Zw = randn(bits, bits);
    [Pw, ~, Qw] = svd(Zw, 'econ');
    Zw = Pw * Qw';
    Gw = W4 - Zw;
    E = ones(col, 1);

    %% Pre-train model
    Ua1 = Wa1 * Xa; Ub1 = Wb1 * Xb;
    Ua2 = Wa2 * Ua1; Ub2 = Wb2 * Ub1;
    H = mu1 * Wa3 * Ua2 + mu2 * Wb3 * Ub2;
    B = sign(W4 * H);

    %% Iterative algorithm
    while (iter < maxIter)

        for ca = 1:d1
            diia(ca) = 1 / (2 * norm(Wa1(ca, :), 2));
        end

        for cb = 1:d1
            diib(cb) = 1 / (2 * norm(Wb1(cb, :), 2));
        end

        Da = diag(diia);
        Db = diag(diib);
        % Update Wa1 & Wb1
        Wa1 = (mu1 * Wa2' * Wa3' * Wa3 * Wa2 + gamma * Da + theta * Wa2' * Wa2) \ (mu1 * Wa2' * Wa3' * H * Xa' + theta * Wa2' * Y * Xa') / ((mu1 + theta) * Xa * Xa' + gamma * eye(row1));
        Wb1 = (mu2 * Wb2' * Wb3' * Wb3 * Wb2 + gamma * Db + theta * Wb2' * Wb2) \ (mu2 * Wb2' * Wb3' * H * Xb' + theta * Wb2' * Y * Xb') / ((mu2 + theta) * Xb * Xb' + gamma * eye(row2));
        % Update Wa2 & Wb2
        Wa2 = (mu1 * Wa3' * Wa3 + (theta + delta) * eye(c)) \ (mu1 * Wa3' * H * Xa' * Wa1' + theta * Y * Xa' * Wa1') / ((mu1 + theta) * Wa1 * Xa * Xa' * Wa1' + delta * eye(d1));
        Wb2 = (mu2 * Wb3' * Wb3 + (theta + delta) * eye(c)) \ (mu2 * Wb3' * H * Xb' * Wb1' + theta * Y * Xb' * Wb1') / ((mu2 + theta) * Wb1 * Xb * Xb' * Wb1' + delta * eye(d1));
        % Update Wa3 & Wb3
        Wa3 = (mu1 * H * Xa' * Wa1' * Wa2') / (mu1 * Wa2 * Wa1 * Xa * Xa' * Wa1' * Wa2' + delta * eye(c));
        Wb3 = (mu2 * H * Xb' * Wb1' * Wb2') / (mu2 * Wb2 * Wb1 * Xb * Xb' * Wb1' * Wb2' + delta * eye(c));
        % Update W4
        Cw = 2 * beta * B * H' - beta * Zw * H * H' + alpha * bits * (2 * (B * Y') * (H * Y')' - (B * E) * (H * E)') - alpha * B * B' * Zw * H * H' + rho * Zw - Gw;
        [Pw, ~, Qw] = svd(Cw, 'econ');
        W4 = Pw * Qw';
        % Update H
        H = ((mu1 + mu2 + beta) * eye(bits) + alpha * W4' * B * B' * W4) \ (mu1 * Wa3 * Wa2 * Wa1 * Xa + mu2 * Wb3 * Wb2 * Wb1 * Xb + alpha * bits * (2 * W4' * B * Y' * Y - W4' * B * E * E') + beta * W4' * B);
        % Update B
        B = sign(2 * alpha * bits * (2 * W4 * H * Y' * Y - W4 * H * E * E') - alpha * W4 * H * H' * W4' * Zb + 2 * beta * W4 * H + rho * Zb - Gb);
        % Update ALM parameters
        Zb = sign(-alpha * W4 * H * H' * W4' * B + rho * B + Gb);
        Cw = -alpha * B * B' * W4 * H * H' - beta * W4 * H * H' + rho * W4 + Gw;
        [Pw, ~, Qw] = svd(Cw, 'econ');
        Zw = Pw * Qw';
        Gb = B - Zb;
        Gw = W4 - Zw;
        % Update weights
        wa1 = 0;
        wb1 = 0;

        for wwwa = 1:c
            wa1 = wa1 + norm(Wa1(wwwa, :), 2);
        end

        for wwwb = 1:c
            wb1 = wb1 + norm(Wb1(wwwb, :), 2);
        end

        h1 = sum(sum((H - Wa3 * Wa2 * Wa1 * Xa).^2)) + theta * sum(sum((Wa2 * Wa1 * Xa - Y).^2)) + delta * (sum(sum((Wa2).^2)) + sum(sum((Wa3).^2))) + gamma * wa1;
        h2 = sum(sum((H - Wb3 * Wb2 * Wb1 * Xb).^2)) + theta * sum(sum((Wb2 * Wb1 * Xb - Y).^2)) + delta * (sum(sum((Wb2).^2)) + sum(sum((Wb3).^2))) + gamma * wb1;
        mu1 = ((1 / h1).^(1 / (t - 1))) / ((1 / h1).^(1 / (t - 1)) + (1 / h2).^(1 / (t - 1)));
        mu2 = ((1 / h2).^(1 / (t - 1))) / ((1 / h1).^(1 / (t - 1)) + (1 / h2).^(1 / (t - 1)));

        % Objective function
        norm1 = mu1 * sum(sum((H - Wa3 * Wa2 * Wa1 * Xa).^2)) + mu2 * sum(sum((H - Wb3 * Wb2 * Wb1 * Xb).^2));
        norm2 = sum(sum((Wa2 * Wa1 * Xa - Y).^2)) + sum(sum((Wb2 * Wb1 * Xb - Y).^2));
        norm3 = sum(sum((B - W4 * H).^2));
        norm4 = sum(sum((bits * (2 * Y' * Y - E * E') - B' * W4 * H).^2));
        norm5 = sum(sum((Wa2).^2)) + sum(sum((Wa3).^2)) + sum(sum((Wb2).^2)) + sum(sum((Wb3).^2));
        norm6a = 0;
        norm6b = 0;

        for rrra = 1:c
            norm6a = norm6a + norm(Wa1(rrra, :), 2);
        end

        for rrrb = 1:c
            norm6b = norm6b + norm(Wb1(rrrb, :), 2);
        end

        norm6 = norm6a + norm6b;
        norm7 = sum(sum((W4 - Zw + (Gw / rho)).^2)) + sum(sum((B - Zb + (Gb / rho)).^2));
        currentF = norm1 + theta * norm2 + beta * norm3 + alpha * norm4 + delta * norm5 + gamma * norm6 + (rho / 2) * norm7;
        fprintf('currentF at iteration %d: %.2f; obj: %.4f\n', iter, currentF, lastF - currentF);

        if (lastF - currentF) < threshold

            if iter > 2
                break
            end

        end

        iter = iter + 1;
        lastF = currentF;
    end

    opt.mu1 = mu1;
    opt.mu2 = mu2;
    opt.Wa1 = Wa1;
    opt.Wb1 = Wb1;
    opt.Wa2 = Wa2;
    opt.Wb2 = Wb2;
    opt.Wa3 = Wa3;
    opt.Wb3 = Wb3;
    opt.W4 = W4;
end
