# **Traffic Sign Recognition** 
[//]: # (Image References)

[image1]: ./examples/all_images.png "Visualization for images in all classes"
[image2]: ./examples/histo.png "Histogram of the train data set"
[image3]: ./examples/transform.png "Image after translation/scaling/rotation" 
[image4]: ./examples/new_images.png "New images"

## Writeup

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

First, I randomly display an image from each traffic sign class. It is shown below.

![alt text][image1]

Next, I print out the histogram of the number of examples per traffic sign class for the training dataset. From the histogram, it's obvious that some classes have much fewer samples than others, e.g. class 0 and class 19 both have fewer than 250 examples while class 2 has about 2000 examples.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

My pre-processing only does normalization. The normalization is (image - 128)/128. In addition, a jittered train dataset is built by adding three versions of transfored train dataset. The transform consist of translation (randomly move in [-2, 2] pixels), scaling (a random [-10%, 10%] ratio), and rotation ([-15, 15] degrees). Thus, the size of the final train dataset becomes 139136. The reason that I didn't add additional images on those classes which contain fewer examples is that I would like to keep this distributation as this might refelcts the real-world distribution. Below shows what an image looks like after appling traslation, scaling, and rotation, respectively.

![alt text][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                                                     | 
|:---------------------:|:-------------------------------------------------------------------:| 
| Input                 | 32x32x3 RGB image                                                   | 
| Convolution 3x3       | 5x5 kernel, 1x1 stride, depth of 6, VALID padding, outputs 28x28x6  |
| RELU                  |                                                                     |
| Max pooling           | 2x2 kernel, 2x2 stride,  outputs 14x14x6                            |
| Convolution 3x3       | 5x5 kernel, 1x1 stride, depth of 6, VALID padding, outputs 10x10x16 |
| RELU                  |                                                                     |
| Max pooling           | 2x2 kernel, 2x2 stride,  outputs 5x5x16                             |
| Fully connected       | inputs 400, outputs 400                                             |
| RELU                  |                                                                     |
| Dropout               | probablity 0.5                                                      |
| Fully connected       | inputs 400, outputs 200                                             |
| RELU                  |                                                                     |
| Dropout               | probablity 0.5                                                      |
| Softmax               | inputs 200, outputs 43                                              |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* The optimizer is AdamOptimizer.
* The batch size is 128. 
* Number of epochs is 30
* The learning rate is 0.0005

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.974
* test set accuracy of 0.962

At first, I implemented the LeNet model (using batch size of 128, number of epochs of 10, and learning rate of 0.001). This model got the test rate accuracy below 0.93 due to overfitting. So, I added the dropout layer using probablity 0.5, and this improves the accuracy on the test set to around 0.93. Next, I changed the hyperparameter setting to the one described above, and also changed the output of the first fully connected layer from 120 to 400 and the output of the second fully connected layer from 84 to 200, this achieves an accuracy around 0.95, but still shows overfitting. Thus, my last resort is to create the jittered dataset. Using this larger size of dataset, the accuracy improves to 0.962.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4]

I don't think there is any difficulities to classify the first five images, however, the perspecitve of the last image not from the front view could make it difficult to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                                 |     Prediction                        | 
|:-------------------------------------:|:-------------------------------------:| 
| Stop Sign                             | Stop sign                             | 
| Go straight or left                   | Go straight or left                   |
| Right-of-way at the next intersection | Right-of-way at the next intersection |
| Speed limit (50km/h)                  | Speed limit (50km/h)                  |
| Speed limit (30km/h)                  | Speed limit (30km/h)                  |
| Speed limit (50km/h)                  | Speed limit (60km/h)                  |

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. The last image is predicted to 60km/h instead of 50km/h which follows my analysis: the not-front-view prespective of the image makes it difficult to classify. This should be able to be solved by adding the other data set by applying affine transfomration. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, very sure this is a stop sign. 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Stop | 1.00000e+00 |
| Yield | 4.20311e-24 |
| No entry | 1.80777e-25 |
| No passing for vehicles over 3.5 metric tons | 3.71291e-27 |
| Speed limit (80km/h)                         | 2.43218e-27 |

For the second image, very sure it is a "go straight or left" sign.

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Go straight or left | 1.00000e+00 |
| Roundabout mandatory | 7.62639e-18 |
| Keep left | 7.89535e-21 |
| Ahead only | 4.93580e-24 |
| Keep right | 5.09918e-26 |

For the third image, very sure it is a "right-of-way at the next intersection" sign. And, "beware of ice/snow" has 0.0000007 probability. These two images does look a bit similar.

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|                 Right-of-way at the next intersection | 9.99999e-01 |
| Beware of ice/snow | 6.35610e-07 |
| Double curve | 1.42321e-18 |
| Children crossing | 3.00282e-20 |
| Pedestrians | 3.25969e-22 |

For the forth image, very sure it is a 50km/h sign. "30km/h" also has a probablity of 0.0000145. To my surprise, "60km/h" is not at top 5 softmax probabilities (in fact it is the top 7, not shown here.), however, the probabilies other than the top 2 are quite small.

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h) |  9.99985e-01 |
| Speed limit (30km/h) |  1.45207e-05 |
| No vehicles | 4.39666e-19 |
| Yield | 3.33982e-19 |
| Speed limit (20km/h) | 6.90084e-20 |

For the fifth image, very sure it is a 30km/h sign. Other speed limit signs are also included in the top 5.

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h) | 1.00000e+00 |
| Speed limit (50km/h) | 1.51344e-08 |
| Speed limit (80km/h) | 4.19746e-09 |
| Speed limit (20km/h) | 1.56980e-10 |
| Road work | 1.00604e-11 |

For the sixth image, there is 0.98% sure it is a 60km/h sign, which is wrong. The correct sign "50km/h" has 0.0178 probability, and good thing is that compared to the predictions done on other images, at least the model is not that sure on this image. Some more image transformation (such as affine transformation) is needed to correctly classify this image. 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h) | 9.82196e-01 |
| Speed limit (50km/h) | 1.77992e-02 |
| Speed limit (80km/h) | 5.27697e-06 |
| No passing  | 7.45391e-13 |
| No vehicles | 6.59927e-13 |