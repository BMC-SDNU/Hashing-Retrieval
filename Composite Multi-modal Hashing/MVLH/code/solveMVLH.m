function [U_final, V_final, mu1_final, mu2_final] = solveMVLH(phi_x1, phi_x2, bits, param)
%% random initialization
[row, col] = size(phi_x1);
[rowt, ~] = size(phi_x2);
threshold = 0.00001;
lastF = 10000000;
iter = 1;

lambda = param.lambda;
t = param.t;
%% Initialize
V = sign(-1+(1-(-1))*rand(bits, col));
U1 = rand(row, bits);
U2 = rand(rowt, bits);
mu1 = 0.5;
mu2 = 0.5;
Phi_X = [sqrt(mu1)*phi_x1', sqrt(mu2)*phi_x2']';
U = [sqrt(mu1)*U1', sqrt(mu2)*U2']';

%% Iterative algorithm
while (iter<100)
    Phi_X = [sqrt(mu1)*phi_x1', sqrt(mu2)*phi_x2']';
    %Update U
    U = (Phi_X*V') / (V*V'+lambda*eye(bits));
    %Update V
    for k = 1:size(V, 1)
        Vk = V; Vk(k,:) = [];
        Uk = U; Uk(:,k) = [];
        V(k,:) = sign((Phi_X'-Vk'*Uk')*U(:,k));
    end
    %Update mu
    U1 = U(1:row, :) / sqrt(mu1);
    U2 = U(row+1:end, :) / sqrt(mu2);
    h1 = sum(sum((phi_x1-U1*V).^2)) + lambda*sum(sum((U1).^2));
    h2 = sum(sum((phi_x2-U2*V).^2)) + lambda*sum(sum((U2).^2));
    mu1 = ((1/h1).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1)));
    mu2 = ((1/h2).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1)));
    
    % compute objective function    
    norm1 = sum(sum((Phi_X-U*V).^2));
    norm2 = sum(sum((U).^2));
    currentF = norm1 + lambda*norm2;
%     fprintf('\ncurrentF at iteration %d: %.2f; obj: %.4f\n', ...
%         iter, currentF, lastF - currentF);
%     if (abs(lastF - currentF) < threshold) 
%         if iter > 1
%             return;
%         end
%     end

    U_final = U;
    V_final = V;
    mu1_final = mu1;
    mu2_final = mu2;
   if (lastF-currentF)<threshold
    return
   end
    iter = iter + 1;
    lastF = currentF;
end
return;

