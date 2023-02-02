function [pre, rec, map] = get_PR_top(Precision, Recall, mAP)
    pre = zeros(1, 20);
    rec = zeros(1, 20);
    map = zeros(1, 20); 
    for i=50:50:1000
        pre(floor(i/50)) = Precision(i);
        rec(floor(i/50)) = Recall(i);
        map(floor(i/50)) = mAP(i);
    end
end