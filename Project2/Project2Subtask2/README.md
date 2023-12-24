# Project 2 Subtask2
<p style="color: red;">DDL: 22:00, Dec.8th</p>

The subtask will be checked in this lab class or the next lab class(before **Dec.8th**) by teachers or SAs. 


## Requirement
```
.
├── README.md: read me first
├── data
│   ├── image_retrieval_repository_data.pkl: an image repository containing image features
│   └── image_retrieval_test_data.pkl: image features of the test set
├── image_retrieval_demo.ipynb: the baseline program of project2 subtask 2
├── NNS.py: implement an NNS model
└── util.py: some common functions

```
You are required to independently train a model using the provided image repository. This model should then be utilized to respond to image queries from the test set. For each image in the test set, should find **5** similar images in the image repository. The response will be in the form of a list of image IDs retrieved from the image repository. The resulting answers must be saved in `retrieval_results.pkl` file. To exemplify this procedure, a Python script named image_retrieval_demo.ipynb has been supplied. You can run `image_retrieval_demo.ipynb` and observe the results.
Additionally, you have the freedom to customize the provided demo by incorporating your own methodologies.

## Grading
When checking, you need to send `retrieval_results.pkl` to the teacher or SA.
For each test image, suppose you submit n similar images(In this subtask, n = 5) and there are m images that are considered to be similar by the evaluation process, the accuracy should be m/n×100%. The test accuracy is the average accuracy of the whole test set.
<p style="color: red;">If the test accuracy of your algorithm exceeds that of the given baseline, you will get all the points for the subtask, otherwise you get a score of 0 for the subtask.
All the code should be zipped and submitted to Blackboard for reproducing the submission and code duplication check.
IF THE CODE IS MISSING, NO POINTS WILL ASSIGNED TO YOU.</p>
