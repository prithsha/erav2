## Code structure and details

Used three normalization techniques
1. Batch normalization : Model file batchNormalNeuralNetwork.py, class: BatchNormalNeuralNetwork
2. Group normalization : Model file groupNormalizationNeuralNetwork.py, class: GroupNormalizationNeuralNetwork
3. Layer normalization : Model file layerNormalizationNeuralNetwork.py, class: LayerNormalizationNeuralNetwork

Here are details of few helper files used by code

1. datasetProvider.py
    - Provide functions to get instances of dataset, transforms and dataloaders 

2. imageVisualizationHelper.py
    - Provide functions to display images

3. testLoopHelper.py
    - Provide functions to execute test_loop of NN

4. trainLoopHelper.py
    - Provide functions to execute training_loop of NN


## Results

### Batch normalization : assignment_8_model_1.ipynb

- Training accuracy:79.30%
- Test accuracy:77.39%
- Learning rate: 0.01
- Batch size: 128
- Epoch: 15
- Parameters: 40544
  
### Group normalization : assignment_8_model_2.ipynb
- 
- Training accuracy:74.29%
- Test accuracy:72.09%
- Learning rate: 0.01
- Batch size: 128
- Group size: 4
- Epoch: 15
- Parameters: 40544. Numbers exactly same as BN
  
### Layer normalization : assignment_8_model_3.ipynb
- 
- Training accuracy:71.78%
- Test accuracy:72.09%
- Learning rate: 0.01
- Batch size: 128
- Epoch: 15
- Parameters: 154520. Large number of parameters compared to BN


