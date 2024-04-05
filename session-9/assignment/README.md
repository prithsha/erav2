## Code structure and details

Used following data augmentation techniques using library albumentations. [Source](./datasetProvider.py)
- horizontal flip 
- shift scale and rotate : (0.1, 0.1, 10)
- radom crop using CoarseDropout (max hight, width =10 , p = 0.5)

### Neural network

Network is designed in file: [dilationNeuralNetwork.py](./dilationNeuralNetwork.py) with class name DilationNeuralNetwork

Every block gets created using class Basic Block which have three fixed layers. Layer configuration changes based on passed parameters. 

1. Layer-1: padding = 0 (Fixed). Can use depth wise convolution 
2. Layer-2: padding = 1 (Fixed). Can use depth wise convolution
3. Layer-3: padding = 1 (Fixed). Can use depth wise convolution or Dilation. Possible to apply stride = 2

Here are details four blocks:
1. Block-1: 
    - Channel in -> out : 3-> 64.  last_layer_stride = 2 . Output : # RF-7, O-15
    - Transition layer: Channel in -> out : 64-> 32.

2. Block-2: 
    - Dilation calculation : RF = (Rin - 1) * D + K + 1
    - Channel in -> out : 32-> 64.  uses dilation at last layer. Output : # RF-32, O-11
    - Transition layer: Channel in -> out : 64-> 32.

3. Block-3: 
    - Channel in -> out : 32-> 64.  uses depth wise convolution. last layer stride = 2. Output : # RF-38, O-5
    - Transition layer: Channel in -> out : 64-> 32.


3. Block-4: 
    - Channel in -> out : 32-> 32.  uses depth wise convolution. last layer stride = 2. Output : # RF-50, O-2


## Results

- Final RF : 50
- Model Parameters count: 191,610
- Train accuracy  : 83.66 %
- Test accuracy   : 85.26 %
- Epochs: 30




