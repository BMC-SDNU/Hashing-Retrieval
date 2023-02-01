### The source code of "Teacher-Student Learning: Efficient Hierarchical Message Aggregation Hashing for Cross-Modal Retrieval"

#### 0. Prepare

Our training environment(only as reference):
    
    Python 3.6.2
    Pytorch 1.5.0
    RTX2080Ti + CUDA 10.2

#### 1. Basic demo

You can easily train the model of our method with the following code:

    chmod a+x ./train.sh
    ./train.sh

#### 2. Using the model on your dataset

(1) Firstly, you need to divide the dataset into three parts: train, query, retrieval. Each part should contain image features, text features and labels.

(2) Then, place the partitioned dataset in the '/Data' directory.

(3) Finally, change the corresponding parameters and train the model.

#### 3. Experimental results on MS COCO

You can easily reimplement the experimental results using the above code. The dataset link is provided in Section.4. If the results you reimplement differ from the below results, it may be due to machine differences. The performances on MS COCO are shown as follow, and the other results can refer our paper.
    
    [16 bits]
        mAP@500 on I2T: 0.691
        mAP@500 on T2I: 0.800
    
    [32bits]
        mAP@500 on I2T: 0.732
        mAP@500 on T2I: 0.869
    
    [64bits]
        mAP@500 on I2T: 0.763
        mAP@500 on T2I: 0.904
    
    [128bits]
        mAP@500 on I2T: 0.790
        mAP@500 on T2I: 0.934
    
    [256bits]
        mAP@500 on I2T: 0.808
        mAP@500 on T2I: 0.947
    
#### 4. Download the dataset and refer the baselines
        
    Google Drive Link:
        https://drive.google.com/drive/folders/1-Ru-NOaIukbJj_wL1rUh7jThOTGwPmKM?usp=sharing
        

If you want to run our code compared with all the cross-modal hashing retrieval baselines on three datasets, we suggest that you should refer the follow link. In this repository, all the datasets are provided and all the baselines can be run easily. You only need change the data path dict "paths_#" and the loaded data key of funtion "load_#()" in "utils/datasets.py".

    https://github.com/BMC-SDNU/Cross-Modal-Hashing-Retrieval
  
  

#### 5. Citation      

    @ARTICLE{HMAH,  
        author={Tan, Wentao and Zhu, Lei and Li, Jingjing and Zhang, Huaxiang and Han, Junwei},  
        journal={IEEE Transactions on Multimedia},   
        title={Teacher-Student Learning: Efficient Hierarchical Message Aggregation Hashing for Cross-Modal Retrieval},   
        year={2022},  
        pages={1-1},  
        doi={10.1109/TMM.2022.3177901}
    }
