This project contains three files that have the logistic regression implementation.

1. "logisticregression.py":
This can be run from the terminal with python three and the desired data file. For example to run this, the command would be:
python3.6 logisticregression.py "data.mat"

This file will run both the training and testing algorithms, and output results of the accuracy. Also it will display loglikelihoods as it iterates.

2. "Logistic-Regression-Classifier.ipynb"
This can be opened as a jupyter notebook and run step by step. Running each box in order will result in running the full logistic regression with gradient descent. It requires python 3 as the kernel, and all the necessary imports. It requires the data files and the classifier.py file to be in the same folder.

3. "classifier.py"
This is a simplified version of "logisticregression.py" containing code meant to be called from the jupyter notebook. This file is imported to be used from the notebook.

Note: first unzip and move out the data files. 