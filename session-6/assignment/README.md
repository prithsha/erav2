# Assignment - 6

## PART-1 Showing back propagation  

Completed excel sheet with formula was created. See following screenshot as reference

![Back-propagation](./Back-propagation-Image-1.png)

### Major steps in calculating impact on weight with respect to loss

1. We need to calculate the partial derivative of loss with respect to each weight
2. Some of the derivatives are know like derivative of sigmoid function
3. Based on known values of derivative we calculated the equation for partial derivative of each weight w.r.t. total-loss 

Screen shot for different learning rates are shown below

Learning rate: 0.1
![Learning rate : 0.1](./0.1-learning-rate.png)

Learning rate: 0.2
![Learning rate : 0.2](./0.2-learning-rate.png)

Learning rate: 0.5
![Learning rate : 0.5](./0.5-learning-rate.png)

Learning rate: 0.8
![Learning rate : 0.8](./0.8-learning-rate.png)

Learning rate: 1.0
![Learning rate : 1.0](./1.0-learning-rate.png)

Learning rate: 2.0
![Learning rate : 2.0](./2.0-learning-rate.png)


## PART-2 Showing back propagation  
Result code file : assignment_6_solution.ipynb

Able to reach accuracy of greater than 99.4% after epoch 17.


Results:

Parameters count:
================================================================
Total params: 13,808
Trainable params: 13,808
Non-trainable params: 0
----------------------------------------------------------------
Log from code file:

Epoch 20

Train: Loss=0.0304 Batch_id=468 Accuracy=99.26: 100%|██████████| 469/469 [00:22<00:00, 21.25it/s]

Test set: Average loss: 0.0155, Accuracy: 9947/10000 (99.47%)

Analysis:
- Consistent accuracy above 99.4% after epoch 17
- Model is under fitting
- Image augmentation helped in increasing test accuracy


Tried different dropout mechanisim like:

1. Added constant drop out in all conv layers 
2. Added batch normalization in every conv layer except max pooling
3. Tried different Model classes. Net4 is final class which is getting used
4. Changed learning rate and momentum