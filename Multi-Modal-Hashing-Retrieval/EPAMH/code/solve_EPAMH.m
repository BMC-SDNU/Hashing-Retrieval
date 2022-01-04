function [R1, R2, mu1, mu2] = solve_EPAMH(V1, V2, n_iter)
    % V1           the 1st modality
    % V2           the 2nd modality
    % n_iter       the number of iterations
    % B            hash code
    threshold = 0.00000001;
    lastF = 100000;
    mu1 = 0.5;
    mu2 = 0.5;

    %% matrix initialization
    [n, bit] = size(V1);
    UX = randn(n, bit);

    R1 = randn(bit, bit);
    [U11 S2 V22] = svd(R1);
    R1 = U11(:, 1:bit);

    R2 = randn(bit, bit);
    [U11 S2 V22] = svd(R2);
    R2 = U11(:, 1:bit);

    %% Iterative algorithm
    for iter = 0:n_iter
        currentF = (1 / mu1) * sum(sum((UX - V1 * R1).^2)) + (1 / mu2) * sum(sum((UX - V2 * R2).^2));
        fprintf('Training: currentF at iteration %.7d: %.5f\n', iter, currentF);

        if (lastF - currentF) < threshold

            if iter > 3
                break
            end

        end

        iter = iter + 1;
        lastF = currentF;

        % update R1
        C1 = UX' * V1;
        [UB1, sigma, UA1] = svd(C1);
        R1 = UA1 * UB1';

        % update R2
        C2 = UX' * V2;
        [UB2, sigma, UA2] = svd(C2);
        R2 = UA2 * UB2';

        % update mu
        mu1 = norm(UX - V1 * R1, 'fro');
        mu2 = norm(UX - V2 * R2, 'fro');

        % update B
        Z = ((1 / mu1) * V1 * R1) + ((1 / mu2) * V2 * R2);
        UX = ones(size(Z, 1), size(Z, 2)) .* -1;
        UX(Z >= 0) = 1;
    end
