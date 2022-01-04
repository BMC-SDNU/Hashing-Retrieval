function map= evaluationMAP(labelgt, results)

% labelgt is the groundtruth, and it is n*c
% results is the labeling results, it is n*c


[trainNum, classNum] = size(results);

map = zeros(1,classNum);
% find the n labels in the results, calculate the precision and recall
parfor i=1:classNum
    rightNum=0;
        
    for j=1:trainNum
        flag = find(labelgt(:, i)==results(j,i));
        if flag>0
            rightNum = rightNum+1;
            map(1,i) = map(1,i) + rightNum/j;
        end
    end 
    map(1,i) = map(1,i)/length(labelgt(:, i));
end