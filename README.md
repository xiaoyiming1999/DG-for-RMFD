# Domain Generalization Repository for Rotating Machinery Fault Diagnosis from HNU Intelligent Fault Diagnosis Group
### Description of DG methods for RMFD
There are eight typical DG methods for RMFD, including AGG, a baseline method that directly aggregates all source-domain data to train a model; four domain alignment methods, DANN, MMD, CORAL, and triplet loss; a data augmentation method, multi-domain Mixup; a meta-learning method, MLDG; and an ensemble learning method, DAEL. Since Our repository is used to process one-dimensional signals, so we made some changes on the model structure and hyperparameters of the methods compared with the original settings. The detailed information on these methods can be found in

|Method              |URL
|------------------- |-------------------------------------------------------------------|
|AGG                 |-
|DANN                |https://arxiv.org/abs/1912.12528                                   |
|MMD                 |https://arxiv.org/abs/1912.12528                                   |
|CORAL               |https://arxiv.org/abs/1912.12528                                   |
|Triplet loss        |https://www.sciencedirect.com/science/article/pii/S0888327022001686|
|multi-domain Mixup  |https://ieeexplore.ieee.org/document/9053273                       |
|MLDG                |https://arxiv.org/abs/1710.03463                                   |
|DAEL                |https://ieeexplore.ieee.org/document/9540778                       |

### Datasets

- **[PHM 2009](https://www.phmsociety.org/competition/PHM/09/apparatus)**
- **[PU Bearing Dataset](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/)**

### Results

|PHM2009|AGG  |DANN |MMD  |CORAL|Triplet loss|multi-domain Mixup|MLDG |DAEL |
|-------|-----|-----|-----|-----|------------|------------------|-----|-----|
|T0     |72.8%|75.1%|73.8%|73.0%|73.5%       |69.2%             |73.4%|59.9%|
|T1     |89.5%|88.4%|89.5%|90.4%|87.7%       |91.7%             |90.0%|92.2%|
|T2     |90.8%|90.0%|90.5%|91.3%|87.2%       |92.1%             |91.1%|92.3%|
|T3     |79.9%|80.4%|79.9%|80.2%|79.1%       |82.6%             |79.3%|68.1%|

|PU     |AGG  |DANN |MMD  |CORAL|Triplet loss|multi-domain Mixup|MLDG |DAEL |
|-------|-----|-----|-----|-----|------------|------------------|-----|-----|
|T0     |35.1%|34.5%|35.8%|38.9%|38.4%       |47.1%             |35.5%|36.2%|
|T1     |90.4%|88.3%|88.7%|91.1%|87.9%       |88.3%             |88.5%|80.0%|
|T2     |45.9%|43.9%|44.3%|45.0%|42.9%       |48.7%             |44.5%|35.3%|
|T3     |88.8%|86.8%|87.8%|88.4%|86.4%       |83.5%             |88.6%|73.5%|

### Description of key parameters

```
--data_name: str, the dataset used
--transfer_task: list, the working conditions of the source and target domains, the first element in the list is the source domain and the second element is the target domain. Note that this code can only implement the case where there are 3 source domains
--normalizetype: str, the data pre-processing method used
--in_channel: int, the number of input channels, usually 1 if data from multiple sensors is not used
--num_train_samples: int, the number of training samples
--num_test_samples: int, the number of test samples
--method: str, the DG method used
--lr: float, the initial learning rate
--inner_lr: float, the inner learning rate for MLDG
--trade_off: float, the trade-off parameter set in the loss function of each method
--gamma: float, the multiplicity of learning rate decay
--steps: str, the epoch at which the learning rate decays
--epoch: int, the number of total epochs
```

### Pakages

This repository is organized as:
- [datasets](https://github.com/xiaoyiming1999/DG-for-RMFD/tree/main/datasets) contains the data augmentation methods and the Pytorch datasets for time domains.
- [models](https://github.com/xiaoyiming1999/DG-for-RMFD/tree/main/models) contains the models used in this project.
- [utils](https://github.com/xiaoyiming1999/DG-for-RMFD/tree/main/utils) contains the functions for realization of the training procedure.

### References

Part of the code refers to the following open source code:
- [UDTL](https://github.com/ZhaoZhibin/UDTL) from the paper "[Applications of Unsupervised Deep Transfer Learning to Intelligent Fault Diagnosis: A Survey and Comparative Study](https://ieeexplore.ieee.org/document/9552620)" proposed by Zhao et al.

### Citiation

If this article inspired you, please cite

```
@article{title={Domain generalization for rotating machinery fault diagnosis: A survey},
        author={Xiao Yiming and Shao Haidong and Yan Shen and Wang Jie and Peng Ying and Liu Bin},
        journal={Advanced Engineering Informatics},
        year={2025}}
```

If you have used the code of our repository, please star it, thank you very much.

### Contact

If you have any questions about the codes or would like to communicate about intelligent fault diagnosis, fault detection, please contact us.

xiaoym@hnu.edu.cn

Mentor E-mail：hdshao@hnu.edu.cn
