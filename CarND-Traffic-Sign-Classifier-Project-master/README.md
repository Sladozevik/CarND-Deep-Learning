#**Traffic Sign Recognition** 

##Writeup Ante Sladojevic

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image3]: ./examples/gray_scale.jpg "Gray Scale"
[image4]: ./new_signs/2.png "Traffic Sign 1"
[image5]: ./new_signs/6.png "Traffic Sign 2"
[image6]: ./new_signs/priority_road_12.png "Traffic Sign 3"
[image7]: ./new_signs/road_work_25.jpg "Traffic Sign 4"
[image8]: ./new_signs/road_work_25_1.jpg "Traffic Sign 5"
[image9]: ./new_signs/school.jpg "Traffic Sign 6"
[image10]: ./new_signs/speed-limit-50_2.png "Traffic Sign 7"
[image11]: ./new_signs/stop.jpg "Traffic Sign 8"
[image12]: ./new_signs/stop_2.jpg "Traffic Sign 9"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!  Also you can find project code in file in submitted zip file traffic-sign-data.zip 

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the 2nd code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 3rd and 4th code cell of the IPython notebook.  

Here is an exploratory visualization of the data set.


Random images with labels.
![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 5th code cell of the IPython notebook.
In seventh code cell there is visualization of images after pre processing. 

As a first step, I decided to convert the images to grayscale because it reduces amount of data and 
network can learn faster.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

After that i used Histogram Equalization because it enhances contrast and removes effect of brightness
After Histogram Equalization I zero centered image data (Mean subtraction)
I completed pre processing by normalizing the image data so the data dimensions are of approximately the same scale.
Scale of images are:

X Train normalize 1.87209831993
X Test normalize 1.86566802053

During gray scale and histogram equalization process depth axes in data have been lost. I needed to return it with newaxis function (from numpy import newaxis).

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Data was provided splitted to Train, Validation and Test set.
I did not add additional data since i wanted to see how my model is working in with defined dataset.
Before build model i shuffle data.

Data information:
X_train shape (34799, 32, 32, 3) #(number of images, size, size, depth)

y_train shape (34799,) # (number of images)

X_test shape (12630, 32, 32, 3)

y_test shape (12630,)

X_valid shape (4410, 32, 32, 3)

y_valid shape (4410,)

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is in the 8th cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GRY image   							| 
| Convolution 7x7     	| 1x1 stride, same padding, outputs 26x26x100 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 13x13x100 				|
| Input         		| 32x32x3 RGB image   							| 
| Convolution 4x4     	| 1x1 stride, same padding, outputs 10x10x150 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x150 			     	|
| Fully connected		| Input 3750 output 250							|
| Dropout        		|                   							|
| Fully connected		| Input 250 output 200							|
| Fully connected		| Input 200 output 43							|
  
Supporting information:

Shapes of Convolutional layers:

Conv 1 shape: (?, 13, 13, 100)

Conv 2 shape: (?, 5, 5, 150)

fc0 shape: (?, 3750)

fc1 shape: (?, 200)

fc3 sahoe: (?, 43)

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is in the 10th cell of the ipython notebook. 

To train the model, I used an Adam optimizer (already implemented in the LeNet lab).
Final settings used were:
batch size: 196

epochs: 100

learning rate: 0.001

mu: 0

sigma: 0.1

dropout keep probability: 0.5

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 11th cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 95%
* test set accuracy of 94%

I started with Lenet network and playing with Pre processing.
I tried Gray Scale, Mean substraction (Zero Centering) and normalization. Validation accuracy was always around 90%.
This was not enough and i need to change stuff.
I read and worked on ImageNet Classification with Deep Convolutional Neural Networks from Alex Krizhevsky - AlexNet but it was to complex for me.
Then i tried Multi-Column Deep Neural Network for Traffic Sign Classifcation (MCDNN) and noticed that i got validation accuracy around 93%.
After that i preprocessed images as in MCDNN (only Gray scale, Histogram) and removed 3rd Convolutional layer due it was reducing images to 1x1 size which i felt was to small for Network. (pixel by pixel).
it was to small for network and from my understanding in picture size 32x32 you can not get much info from layer of 1x1 pixel. 
I also added Dropout in 1st fully connected layer and this proved good and Validation accuracy jumped to 95%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

Answer:LeNet since it was easy :)

* What were some problems with the initial architecture?

Answer: accuracy did not increase, there was no dropout, to big and complex network, i felt whole DLL was suffering. When reading many blogs and research papers i seen that more simple DLL works well.


* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Answer: See above explanation

* Which parameters were tuned? How were they adjusted and why?

Answer: Honestly i played with all parameters and changed all layers. After that i started to adjust network according to how i noticed accuracy is changing. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Answer: I used CPU for training so i noticed that i need to test my model on lower number of EPOCHS (5-10). If model turned out well i used 100 EPOCH allnighters :)
I noticed that in many cases there was over fitting that is why i added in first fully connected layer Dropout. I used 0.5 but later on when i played it would make sense to to start with 1.0.

If a well known architecture was chosen:
* What architecture was chosen?

Answer: As Described above i used MCDNN

* Why did you believe it would be relevant to the traffic sign application?

Answer: It was simple, it had good preprocessing data and it could be easily adjusted. 
Reading MCDNN research paper i felt is best network to work with.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

Answer: As you can see Validation accuracy is 95% and Test accuracy is 94%. this means there is no big overfitting and network works well


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine different traffic signs that I found on the web.

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11] ![alt text][image12]

All images are different sizes and this was issue in beginning so i needed to reduce size.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h       		| 30 km/h   									| 
| Turn Left     		| Turn Left										|
| Priority road			| Priority road									|
| Road work	      		| Road work 					 				|
| Road work	      		| Road work 					 				|
| Children crossing		| No passing         							|
| 50 km/h       		| Turn Right    								| 
| Stop Sign      		| Priority road									| 
| Stop Sign      		| Stop sign   									| 


The model was able to correctly guess 6 of the 9 traffic signs, which gives an accuracy of 67%. 
I am not satisfied with accuracy of it. Cleary model can be better. 
Strangely most clear images sign 50km/h,Stop,Children crossing it did not accurately predict. I feel this is due preprocessing.
What i would do to improve model performance and accuracy is: Augment the Training Data, fallow step by step MCDNN pre processing. 

Adding additional explanation:

The accuracy on the new sign images is 67% while it perfomed with 94% on the testing set thus it seems the model is overfitting.
Model did not do well on new data. I see that clear images sign 50km/h,Stop,Children crossing model did not reckognize.
This can be improved by:
* improving preprocessing of images: Augment the Training Data, fallow step by step MCDNN pre processing. 
* incresing number of examples of training images
* improve model adding additional Convolutional Layers and or Dropouts


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.

 The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0            		| 30 km/h   									| 
| 1.0           		| Turn Left										|
| 1.0        			| Priority road									|
| 0.96          		| Road work 					 				|
| 1.0    	      		| Road work 					 				|
| 0%                 	| children crossing        						|
| 0%            		| 50 km/h    						     		| 
| 0%             		| Stop Sign								        | 
| 1.0           		| Stop sign   									| 

As you can see it did not at all correctly predict Children crossing, 50km/h and Stop sign. 