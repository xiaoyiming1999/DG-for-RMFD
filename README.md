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

### Contact

If you have any questions about the codes or would like to communicate about intelligent fault diagnosis, fault detection, please contact us.

xiaoym@hnu.edu.cn

Mentor E-mail：hdshao@hnu.edu.cn
