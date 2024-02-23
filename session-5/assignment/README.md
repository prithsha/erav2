# Creating first neural network


## Source code structure
- s5.ipyb: starting notebook file. File references other helper files as defined below
- model.py: Model is defined in this
- utils.py: A utility module , which provided common methods for training and testing models
- constant.py: Any constant used in project
- loggingInit.py: Initializes logging configuration.
- application.log: Log file generated during execution


## How to execute the code
In constant.py, you can define the location where downloaded data should be stored.
1. Install required dependencies
2. Start executing s5.ipynb

## Program flow

1. setup logging and get device information where it is going to execute (cpu/gpu)
2.  Create train and test transform 
3. Using above transform create instances of train and test data. This will download data, if data is not available in designated folder
4. create train and test loader which helps in enumerating data
5. Check data visually for a batch
6. Create the instance of created network and select optimizer and loss function 
6. send data for training and record losses and accuracy 
7. Once model gets executed for a batch send it for testing and record accuracy and losses
8. repeat above two steps multiple times and check how accuracy and losses are changing over time.

