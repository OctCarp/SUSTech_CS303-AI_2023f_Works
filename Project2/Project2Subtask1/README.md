#Project 2 Subtask1
<p style="color: red;">DDL: 22:00, Nov.24th</p>

The subtask will be checked in this lab class or the next lab class(before **Nov.24th**) by teachers or SAs. 

## Environment preparation
Python version: 3.10.8

Packages:
```
matplotlib==3.7.1
numpy==1.26.1
tqdm==4.64.1
```
You can use `pip install -r requirements.txt` to install the above packages.

It is recommended to create a separate environment to keep your programs of project2 isolated from project1.
```
conda create -n python3108 python=3.10.8
conda activate python3108
conda install jupyter
pip install -r requirements.txt
```

## Requirement
```
├── README.md: read me first
├── requirements.txt: packages needed
├── data
│   ├── classification_test_data.pkl: image features of the test set
│   ├── classification_train_data.pkl: image features of the training set
│   └── classification_train_label.pkl: image corresponding labels of the training set
├── image_classification_demo.ipynb: the baseline program of project1 subtask 1
├── SoftmaxRegression.py:implement a softmax regression model
└── util.py: some common functions
```
You are required to independently train a model using the provided training set. This model should then be utilized to predict labels for the test set. The predicted labels are to be saved in  `classification_results.pkl` file. To exemplify this procedure, a Python script named image_classification_demo.ipynb has been supplied. You can run `image_classification_demo.ipynb` and observe the results.
Additionally, you have the freedom to customize the provided demo by incorporating your own methodologies.

## Grading

When checking, you need to send `classification_results.pkl` to the teacher or SA.
<p style="color: red;">If the test accuracy of your algorithm exceeds that of the given baseline, you will get all the points for the subtask, otherwise you get a score of 0 for the subtask.
All the code should be zipped and submitted to blackboard for reproducing the submission and code duplication check.
IF THE CODE IS MISSING, NO POINTS WILL ASSIGNED TO YOU.</p>
