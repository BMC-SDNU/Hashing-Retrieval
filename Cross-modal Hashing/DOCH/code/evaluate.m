function [Wx, Wy] = evaluate(XChunk,YChunk,LChunk,XTest,YTest,LTest,param)

    for chunki = 1:param.nchunks
        
        LTrain = cell2mat(LChunk(1:chunki,:));
        XTrain_new = XChunk{chunki,:};
        YTrain_new = YChunk{chunki,:};
        LTrain_new = LChunk{chunki,:};
        
        % Hash code learning  

        if chunki == 1
            [Wx,Wy,BB,MM] = train0(XTrain_new,YTrain_new,param,LTrain_new);
        else
            [Wx,Wy,BB,MM] = train(XTrain_new,YTrain_new,param,LChunk,BB,MM,chunki);
        end
        
        B = cell2mat(BB(1:end,1));
        
%         R = size(B,1);
        %% iamge as query to retrieve text database
        BxTest = compactbit(XTest*Wx >= 0);
        ByTrain = compactbit(B >= 0);
        DHamm = hammingDist(BxTest, ByTrain);
        [~, orderH] = sort(DHamm, 2);
        evaluation_info.Image_VS_Text_MAP  = mAP(orderH', LTrain, LTest);
        
        
        %% text as query to retrieve image database
        ByTest = compactbit(YTest*Wy >= 0);
        BxTrain = compactbit(B >= 0);
        DHamm = hammingDist(ByTest, BxTrain);
        [~, orderH] = sort(DHamm, 2);
        evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);
        fprintf('DOCH %d bits -- chunk: %d,   Image_VS_Text_MAP: %f,   Text_VS_Image_MAP: %f \n',param.nbits ,chunki, evaluation_info.Image_VS_Text_MAP, evaluation_info.Text_VS_Image_MAP);
        
        clear evaluation_info
        
    end
    fprintf('\n');
end
