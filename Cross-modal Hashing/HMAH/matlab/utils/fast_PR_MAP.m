function [Precision, Recall, Map] = fast_PR_MAP(cateTrainTest, HammTrainTest)     
    nTrain = size(cateTrainTest, 1);
    nTest = size(cateTrainTest, 2);
    Precision = zeros(1, nTrain);
    Recall = zeros(1, nTrain);
    Map = zeros(1, nTrain);
    truth_pair = sum(cateTrainTest, 1);
    % fprintf('ALL is %d, trueth is %d\n', nTrain*nTest, sum(truth_pair))
    cnt = 0;
    easy_sample = 0;
    for i=1:nTest
        if truth_pair(1, i) == 0
            continue
        end
        cnt = cnt + 1;
        pre = 0;
        temp_P = 0;
        for j=1:nTrain
            if cateTrainTest(HammTrainTest(j, i), i) == 1
                pre = pre + 1;
                temp_P = temp_P + pre/j;
            end
            % if j == 1000
            %     fprintf('[%d] has %d.\n', i, pre);
            % end
            Precision(j) = Precision(j) + pre/j;
            Recall(j) = Recall(j) + pre/truth_pair(i);
            if pre ~= 0
                Map(j) = Map(j) + temp_P / pre;
                % if j == nTrain
                    % fprintf('The sample %d is %.4f\n', i, temp_P / pre);
                    % if (temp_P / pre) > 0.98
                        easy_sample = easy_sample + 1;
                    % end
                % end
            else
                Map(j) = Map(j);
            end
        end
    end
    Precision = Precision / cnt;
    Recall = Recall / cnt;
    Map = Map / cnt;
    % easy_sample
 end