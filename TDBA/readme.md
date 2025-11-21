# TDBA Framework

This repository contains the implementation code for the TDBA

## Dependencies

All required dependencies for this project are listed in the `requirements.txt` file. To install these dependencies, run the following command:

```
pip install -r requirements.txt
```

## Framework Foundation

The main framework of this codebase is built upon the BackTime method. Proper citation to the original BackTime work has been included in the main paper.

## Code Structure

*   `main.py`: Serves as the main entry point of the program. 

*   `attack.py`: Contains functions responsible for implementing the backdoor attack logic, including the core mechanisms of the TDBA framework.

*   `dataset.py`: Provides data processing functionalities, including data loading, preprocessing, and preparation for training and testing.

*   `train.py`: Implements functions for executing the training and testing processes of the forecasting models under both clean and attacked scenarios.

*   `trigger.py`: Includes implementations of the two types of trigger generator involved in this work
