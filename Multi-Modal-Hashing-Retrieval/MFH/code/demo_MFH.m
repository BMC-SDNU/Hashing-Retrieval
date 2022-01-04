function [] = demo_MFH(bits, dataname)
    % delete(gcp);
    % parpool; % train the subspaces in parallel
    % model_types = {'mfh', 'sh', 'itq', 'okmeans', 'kmh'};
    warning off;
    bits = str2num(bits);
    model_types = {'mfh'};
    run = 5;

    for rrr = 1:run
        for i = 1:numel(model_types)
            model_type = model_types{i};
            [P] = MHash(bits, dataname);
            map(rrr) = P;
        end
    end

    fprintf('[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%s] MAP = %.4f\n', dataname, num2str(bits), mean(map));
end
