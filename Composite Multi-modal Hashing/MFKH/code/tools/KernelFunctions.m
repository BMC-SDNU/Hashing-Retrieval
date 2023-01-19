function V = KernelFunctions(kernel_name, Xi, Xj, sigma)

switch kernel_name
    case 'gauss'
        V = distMat(Xj', Xi', 0);
        tal = sum(V(:)/(3000*3000));
        V = exp(-sigma*V);
    case 'chi2'
        V = Chi2Kernel(double(Xi), double(Xj));
        V = exp(-sigma*V);
    case 'sum_min'
        V = zeros(size(Xj, 2), size(Xi, 2));
        for i = 1:size(Xj, 2)
            V(i, :) = sum(bsxfun(@min, Xi, Xj(:, i)));
        end
    otherwise
        
end
