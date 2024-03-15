# Assignment - 7

## Step-1 Model

Included sub-steps:
- Setup
- Basic Skeleton
- Lighter model
- Batch Normalization

File name: assignment_7_model_1.ipynb

Targets:
- Model parameters nearly 7k
- Test accuracy more than 99.2 %
- Setup basic code with transition and max pooling block
- Adding batch normalization in each convolution layer

Results:
- Total params: 6666
- Maximum test accuracy: 99.10%
- Maximum train accuracy : 99.52%

Analysis:
- Model does not seems to be over fitting
- Test accuracy not increasing, infact it started decreasing after epoch-10
- Need to reduce model parameters and kernel size of 7 at last layer

File Link: [Model-1: assignment_7_model_1.ipynb](https://github.com/prithsha/erav2/blob/main/session-7/assignment/assignment_7_model_1.ipynb)  
 
## Step-2 Model

Included sub-steps:
- Dropout
- Added global average polling at the end
- Increases capacity in last few layer and added one additional layer
- No Change in max pooling location

File name: assignment_7_model_2.ipynb

Targets:
- Model parameters under 8k
- Test accuracy more than 99.4 %
- Test accuracy should increase as capacity has increased
- Gap between train and target should reduce

Results:
- Total params: 7744
- Maximum test accuracy: 99.25%
- Maximum train accuracy : 98.81%

Analysis:
- Model does not seems to be over fitting
- On first epoch train accuracy is reduced. No idea why ?
- Overall test accuracy increased but train accuracy decreased.

File Link: [Model-2: assignment_7_model_2.ipynb](https://github.com/prithsha/erav2/blob/main/session-7/assignment/assignment_7_model_2.ipynb) 
 

## Step-3 Model

Included sub-steps:
- Added image random rotation 
- Changed learning rate to see change in accuracy

File name: assignment_7_model_3.ipynb

Targets:
- Model parameters under 8k
- Test accuracy more than 99.4 %
- Test accuracy should increase and initial train accuracy will reduce.
- Gap between train and target should reduce and be consistent
- Change step size , gamma and learning rate in optimizer and scheduler

Results:
- Total params: 7744
- Maximum test accuracy: 99.44
- Maximum train accuracy : 98.96

Analysis:
- Model does not seems to be over fitting
- Overall test accuracy increased.
- Increased learning rate and reduced step size brings train accuracy high in early epoch. 

File Link: [Model-3: assignment_7_model_3.ipynb](https://github.com/prithsha/erav2/blob/main/session-7/assignment/assignment_7_model_3.ipynb) 
 
