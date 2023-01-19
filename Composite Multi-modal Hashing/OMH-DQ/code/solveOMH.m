function [B_db, B_test] = solveOMH(phi_x1, phi_x2, Phi_testI, Phi_testT, Phi_dbI, Phi_dbT, param, j)
% phi_x1     the nonlinearly transformed representation of the 1st modality
% phi_x2     the nonlinearly transformed representation of the 2nd modality
% W1         the projection matrix of the 1st modality
% W2         the projection matrix of the 2nd modality
% H          the consensus multi-modal factor
% mu1        the weight of the 1st modality
% mu2        the weight of the 2nd modality       
% B          hash code 
% R          rotation matrix

    %% setting parameters
    [p, col] = size(phi_x1);
    bits = param.bits;
    maxIter = param.maxIter;
    beta= param.beta;
    alpha = param.alpha;
    delta = param.delta;
    lambda = param.lambda;
    mu1 = 0.5;
    mu2 = 0.5;
    iter = 1;
    lastF = 10000000000;

    %% matrix initialization
    L = param.L;
    B = sign(-1+(1-(-1))*rand(bits, col));
    Zb = sign(-1+(1-(-1))*rand(bits, col));
    Gb = B-Zb;
    R = randn(bits, bits);
    [Pr,~,Qr] = svd(R, 'econ');    
    R = Pr * Qr';
    Zr = randn(bits, bits);
    [Pz,~,Qz] = svd(Zr, 'econ');    
    Zr = Pz * Qz';
    Gr = R-Zr;
    E = ones(col, 1);
    H = (beta*R'*B + alpha*bits*(2*R'*B*L'*L-R'*B*E*E')) / (beta+alpha);
    W1 = ((1/mu1)*H*phi_x1') / ((1/mu1)*phi_x1*phi_x1' + delta*eye(p));
    W2 = ((1/mu2)*H*phi_x2') / ((1/mu2)*phi_x2*phi_x2' + delta*eye(p));

    %% Iterative algorithm
    while (iter<maxIter)     
        % update \mu
        mu1 = sqrt(sum(sum((B-W1*phi_x1).^2))) / (sqrt(sum(sum((B-W1*phi_x1).^2)))+sqrt(sum(sum((B-W2*phi_x2).^2))));
        mu2 = sqrt(sum(sum((B-W2*phi_x2).^2))) / (sqrt(sum(sum((B-W1*phi_x1).^2)))+sqrt(sum(sum((B-W2*phi_x2).^2))));
        % update Wm
        W1 = ((1/mu1)*H*phi_x1') / ((1/mu1)*phi_x1*phi_x1' + delta*eye(p));
        W2 = ((1/mu2)*H*phi_x2') / ((1/mu2)*phi_x2*phi_x2' + delta*eye(p));
        % update R
        Cr = 2*beta*B*H' + 2*alpha*bits*(2*(B*L')*(H*L')'-(B*E)*(H*E)') - alpha*B*B'*Zr*H*H' + lambda*Zr - Gr;
        [Pr,~,Qr] = svd(Cr, 'econ');    
        R = Pr * Qr';   
        % update H
        H = ((1/mu1)*eye(bits)+(1/mu2)*eye(bits)+beta*eye(bits)+alpha*R'*B*B'*R) \ ((1/mu1)*W1*phi_x1 + (1/mu2)*W2*phi_x2 + beta*R'*B + alpha*bits*(2*R'*B*L'*L-R'*B*E*E'));
        % update B
        B = sign(2*beta*R*H + 2*alpha*bits*(2*R*H*L'*L-R*H*E*E') - alpha*R*H*H'*R'*Zb + lambda*Zb - Gb);   
        % update ALM parameters
        Zb = sign(-alpha*R*H*H'*R'*B+lambda*B+Gb);
        Cz = -alpha*B*B'*R*H*H'+lambda*R+Gr;
        [Pz,~,Qz] = svd(Cz, 'econ');    
        Zr = Pz * Qz';
        Gb = B-Zb;
        Gr = R-Zr;

       % objective function    
        norm1 = mu1*sum(sum((B-W1*phi_x1).^2)) + mu2*sum(sum((B-W2*phi_x2).^2));
        norm2 = sum(sum((B-R*H).^2));
        norm3 = sum(sum((bits*(2*L'*L-E*E')-B'*R*H).^2));
        norm4 = sum(sum((W1).^2)) + sum(sum((W2).^2));
        norm5 = sum(sum((R-Zr+(Gr/lambda)).^2)) + sum(sum((B-Zb+(Gb/lambda)).^2));
        currentF = norm1 + beta*norm2 + alpha*norm3 + delta*norm4 + (lambda/2)*norm5;
        fprintf('currentF at iteration %d: %.2f; obj: %.4f\n', iter, currentF, lastF - currentF);  

        iter = iter + 1;
        lastF = currentF;
    end
    fprintf('-------------------Run %d Finished!-------------------\n', j);
    B_tr = B' >0;

    %% Hash function
    W1_fin = W1;
    W2_fin = W2;

    %% Query-adaptive Online Hashing with Dynamic Weights
    % setting parameters
    iter = 1;
    % lastF = 10000000000;
    % matrix initialization
    phi_x1 = Phi_testI;
    phi_x2 = Phi_testT;
    [~, colte] = size(phi_x1);
    B = sign(-1+(1-(-1))*rand(bits, colte));
    % iterative online hashing
    while (iter<5)
        W1 = W1_fin;
        W2 = W2_fin;  
        % update WEIGHT
        mu1 = sqrt(sum(sum((B-W1*phi_x1).^2))) / (sqrt(sum(sum((B-W1*phi_x1).^2)))+sqrt(sum(sum((B-W2*phi_x2).^2))));
        mu2 = sqrt(sum(sum((B-W2*phi_x2).^2))) / (sqrt(sum(sum((B-W1*phi_x1).^2)))+sqrt(sum(sum((B-W2*phi_x2).^2))));
        % update B
        B = sign(2*mu1*W1*phi_x1 + 2*mu2*W2*phi_x2);

    %     % objective function for online hashing
    %     norm1 = mu1*sum(sum((B-W1*phi_x1).^2)) + mu2*sum(sum((B-W2*phi_x2).^2));
    %     norm2 = sum(sum((W1).^2)) + sum(sum((W2).^2));
    %     currentF = norm1 + delta*norm2;
    %     fprintf('currentF at iteration %d: %.4f; obj: %.4f\n', iter, currentF, lastF - currentF);
        iter = iter + 1;
    end
    B_test = B'>0;

    iter =1;
    %lastF = 10000000000;
    phi_x1 = Phi_dbI;
    phi_x2 = Phi_dbT;
    [~, coldb] = size(phi_x1);
    B = sign(-1+(1-(-1))*rand(bits, coldb));
    while (iter<5)
        W1 = W1_fin;
        W2 = W2_fin;
        % update WEIGHT
        mu1 = sqrt(sum(sum((B-W1*phi_x1).^2))) / (sqrt(sum(sum((B-W1*phi_x1).^2)))+sqrt(sum(sum((B-W2*phi_x2).^2))));
        mu2 = sqrt(sum(sum((B-W2*phi_x2).^2))) / (sqrt(sum(sum((B-W1*phi_x1).^2)))+sqrt(sum(sum((B-W2*phi_x2).^2))));
        % update B
        B = sign(2*mu1*W1*phi_x1 + 2*mu2*W2*phi_x2);    

    %     % objective function for online
    %     norm1 = mu1*sum(sum((B-W1*phi_x1).^2)) + mu2*sum(sum((B-W2*phi_x2).^2));
    %     norm2 = sum(sum((W1).^2)) + sum(sum((W2).^2));
    %    currentF = norm1 + delta*norm2;
    %     fprintf('currentF at iteration %d: %.4f; obj: %.4f\n', iter, currentF, lastF - currentF);
        iter = iter + 1;
    end
    B_db = B'>0;
end
