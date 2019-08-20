# Here you will find the results of the experiments conducted for "Loss Std Regularization"
The results are separated in folders by dataset/task. In each such folder, there is a folder for each model which was used. 

Every end folder contains the code for the model, a checkpoint of the best epoch, tsv and diagram files containing metrics from the training process and a README file describing the configuration used to obtain said results. In addition, the folders will contain similar files and descriptions of the baseline performance of said model withouth applying our technique.

So far the following datasets/tasks and models were experimented upon:
- [ ] MNIST
  - [ ] AlexNet
  - [ ] Simple MLP
- [ ] CIFAR-10
  - [ ] VGG-11
  - [ ] VGG-16
  - [ ] VGG-19
  - [ ] ResNet101
  - [ ] ResNeXt
  - [ ] LeNet
  - [ ] AlexNet
- [ ] CIFAR-100
  - [ ] VGG-11
  - [ ] VGG-16
  - [ ] VGG-19
  - [ ] ResNet101
  - [ ] ResNeXt
- [ ] COCO
  - [ ] VGG-11
  - [ ] VGG-16
  - [ ] VGG-19
  - [ ] ResNet101
  - [ ] ResNeXt
- [ ] Sentiment140
    - [ ] #TODO Find popular models for this task
- [ ] Open AI Gym games #TODO Choose a few games to benchmark
    - [ ] #TODO Find popular models for this task