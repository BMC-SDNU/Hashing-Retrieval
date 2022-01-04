function [result] = query_videos(query_id, seeds, idx_stat, idx_end, binaryCode, model)
%%
% This is the query methods using wuxiao's algorithm
% query_id is the video id for query
% seeds are the seed videos seleted

%%
    seed_num = seeds(query_id);
    kf_query_id = idx_stat(seed_num):idx_end(seed_num);
    kf_query = binaryCode(kf_query_id,:);
    
    
    if size(binaryCode, 2)==38
        fprintf('Symmetric quantizer distance (SQD):\n');
        centers = double(cat(3, model.centers{:}));  % Assuming that all of the
        queryR2 = double(reconstruct_by_ckmeans(kf_query', model));
        [index, dist] = linscan_aqd_knn_mex(binaryCode', queryR2, ...
            size(binaryCode, 1), model.nbits, 100000, centers);
        dist = dist'; index = index';
        idx = [];
        for i=1:size(kf_query,1)
            if sum(dist(i,:)==0) >=2000
                idx = [idx i];
            end
        end
        dist(idx,:)=[]; index(idx,:)=[];
        dist = reshape(dist, 1, size(dist,1)*size(dist,2));
        index = reshape(index, 1, size(index,1)*size(index,2));
        [~, idx_kf] = sort(dist);
        idx_kf = index(idx_kf);
        [~,j] = unique(idx_kf,'first');
        idx_kf = idx_kf(sort(j));
    else
        dist = pdist2(kf_query, binaryCode);% Here should be replaced by Hamming distance for speed.
%         kf_query = compactbit(kf_query);binaryCode = compactbit(binaryCode);dist = hammingDist(kf_query, binaryCode);
        idx=[];
        for i=1:size(kf_query,1)
            if length(find(dist(i,:)==0)) >=2000
                idx = [idx i];
            end
        end
        dist(idx,:)=[];
        dist = min(dist);
        %     dist = sort(pdist2(kf_query, binaryCode));
        %     dist = mean(dist(1:1+floor(size(kf_query,1)/5), :));
        [~, idx_kf] = sort(dist);
    end
    
    result = zeros(size(binaryCode,1), 1);
    parfor i=1:min(ceil(size(binaryCode,1)/30), length(idx_kf))
        temp = find((idx_stat<=idx_kf(i)).*(idx_end>=idx_kf(i))==1);
        if isempty(temp)
            temp = 1;
        end
        if mod(i, 10^5)==0
            fprintf('%d:%d\n',i,temp);
        end
        result(i) = temp;
    end
    
    [~,j] = unique(result,'first');
    result = result(sort(j));    