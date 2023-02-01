function [] = curve(nbit, db_name)
%
% Syntax: [] = curve(nbit, db_name)
%
% Long description
    addpath('utils/');
    fprintf('Load hashcode from Our-model.\n');
    save_path = sprintf('../Hashcode/GCN_%d_%s_bits.mat', nbit, db_name);
    load(save_path);
    % query_image_hash, query_txt_hash, retrieval_image_hash, retrieval_txt_hash
    B_trnx = compactbit(retrieval_img_B > 0);
    B_trnt = compactbit(retrieval_txt_B > 0);
    B_tstx = compactbit(val_img_B > 0);
    B_tstt = compactbit(val_txt_B > 0);

    Dhammx_t = hammingDist(B_tstx, B_trnt)';
    Dhammt_x = hammingDist(B_tstt, B_trnx)';

    [~, Dhammx_t_index] = sort(Dhammx_t, 1);
    [~, Dhammt_x_index] = sort(Dhammt_x, 1);

    [Pre_I2T, Rec_I2T, MAP_I2T] = fast_PR_MAP(int32(cateTrainTest), int32(Dhammx_t_index));
    [Pre_T2I, Rec_T2I, MAP_T2I] = fast_PR_MAP(int32(cateTrainTest), int32(Dhammt_x_index));

    [Pre_I2T_top, Rec_I2T_top, MAP_I2T_top] = get_PR_top(Pre_I2T, Rec_I2T, MAP_I2T);
    [Pre_T2I_top, Rec_T2I_top, MAP_T2I_top] = get_PR_top(Pre_T2I, Rec_T2I, MAP_T2I);

    result_name = ['../Result/' db_name '_' num2str(nbit) '_result' '.mat'];
    save(result_name, 'MAP_I2T_top', 'MAP_T2I_top', 'Pre_I2T_top', 'Pre_T2I_top', 'Rec_I2T_top', 'Rec_T2I_top', 'MAP_I2T', 'MAP_T2I', 'Pre_I2T', 'Pre_T2I', 'Rec_I2T', 'Rec_T2I');

    fprintf('MAP_I2T@All=%.3g MAP_T2I@All=%.3g TOPK@500_I2T=%.3g TOPK@500_T2I=%.3g \n', MAP_I2T(end), MAP_T2I(end), MAP_I2T(500), MAP_T2I(500));
    fprintf('\n');

end
