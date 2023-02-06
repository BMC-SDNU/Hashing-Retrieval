# Creating-Something-from-Nothing-Unsupervised-Knowledge-Distillation-for-Cross-Modal-Hashing
The source code for the CVPR2020 paper "Creating Something from Nothing: Unsupervised Knowledge Distillation for Cross-Modal Hashing". 
The paper is avaliable in "https://arxiv.org/abs/2004.00280". 

![Image Text](https://github.com/huhengtong/UKD/blob/master/framework.png)  

This codes are based on the third-party codes for UGACH(https://github.com/PKU-ICST-MIPL/UGACH_AAAI2018) and SSAH(https://github.com/zyfsa/cvpr2018-SSAH). 

The codes consist of two parts, namely teacher model and student model. Training the teacher model firstly to obtain the features of different modality data, then estimating similarities by calculating the distance between features. Finally, using the similarity information to supervise the learning of the student model. 

The experiments are conducted on MIRFlickr (link:https://pan.baidu.com/s/1IkT7x9XSgKr-V7_LigxTEQ  password:w4my) and NUS-WIDE (link:https://pan.baidu.com/s/1wN-de-eIqrjNQ72N8ZA5jA  password:7x8y). The codes for data processing are available in teacher_model/KNN.  
Requirements:
  Python 3; tensorflow

To train the teacher model, running teacher_train.py in the file of teacher_model;  
To calculate the similarity information, running calcu_sim.py in the file of teacher_model;  
To train the student model, running Main.py in the file of student_model;  
