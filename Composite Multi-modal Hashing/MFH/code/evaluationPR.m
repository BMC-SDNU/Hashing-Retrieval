function precision =  evaluationPR(gt, result)

% ft is n*1
% groundTruth is n*1
% precision=[0.1, ..., 1], like this
% 
precision = zeros(1,10);

rightNum=0;
tokens = floor(length(gt)*(0.1:0.1:1));

for j=1:length(result)
    flag=find(gt==result(j));
    if flag>0
        rightNum=rightNum+1;
        toIdx = find(tokens==rightNum);
        if toIdx>0
            precision(toIdx)=rightNum/j;
        end
    end
end
precision(10)=rightNum/j;
