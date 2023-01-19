function topN=TOPK(hammTrainTest,feaTrain_class,feaTest_class,nbits)  %multi-label
fprintf('compute the performance\n');
% nbits = 64;
disp(nbits);
tmp_mat = feaTest_class*feaTrain_class';%ï¿½ï¿½ï¿?
rel_mat = tmp_mat>=1;

[sort_val, sort_idx]= sort(hammTrainTest, 1, 'ascend');
for i = 1:size(hammTrainTest, 2)
    qry_label = feaTest_class(i) ;
    database_label = rel_mat(i,:);
    ret_label = database_label(sort_idx(:,i));
    for j = 1000:1000:10000
        precision(i, j/1000) = length(find(ret_label(1:j)==1))/j ;
        recall(i, j/1000)    = length(find(ret_label(1:j)==1))/(sum(ret_label)+eps);
    end
    end
topN = mean(precision);
end