# Predicting Steering Angles in Driving Simulator using Deep Learning

## Synopsis

Successfully navigating an automobile through a simulator using a Convolutional Neural Network (CNN) involved significant experimentation and research. The process which produced this model could be summarized in seven steps. The first was to use a smaller, ideally faster, and already developed CNN in order to immediately measure performance. The second was to process images from the simulator before they were passed to the simulator. The third step involved building a generator in order for the model to use many more images without encountering memory limitations. The fourth, was to augment images increasing the maximum number and vairety of images that could be used in training. The fifth, involved tunning parameters and selecting data appropriately. The sixth involved improving the model and using a deeper CNN with more layers. Finally, the seventh step was to take precautions to avoid the overfitting of the data. While other solutions exist and many additional steps could be implemented these seven steps are a good starting point especially if done in order.

## Training Strategy
A strategic approach emerged closer to the middle of the project as completing these steps in a different order produced challenges. For instance, a large model with maxpooling was used first and training time was very long. Also, augmentations were attempted before a generator was completed and could not be used until the generator worked. Posts by others completing the project and the academic paper by [Nvidia researchers](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) was most helpful in developing a strategy.

### 1. First Model
The [Commaii CNN](https://github.com/commaai/research/blob/master/train_steering_model.py) made up of 5 layers was used first. They were made up of three convolutional layes with large patch and stride sizes, a flatten layer, and two dense layers each with dropouts. It used an Adaptive Moment Estimation (Adam) optimizer thereby computing adapting learning rates for each parameter while keeping an exponentially decaying average of past gradients. Rectified Linear Units (Relus) and Exponential Linear Units (Elus) activations were both used but did not noteably impact performance or training speed. It is trained to minimize the Mean Squared Error (MSE).

Using this model training and unaltered images training took several hours and the vehicle was unable to navigate the first turn of the track. The vehicle wobbled slightly, went off the road and then went up a hill. 

### 2. Prepocessing
The following image preprocessing steps led to a much faster training time and a slight improvement on the the cars navigation of the first corner.

- the height of the image was cropped to remove the horizon and the dash of the car
- the image was resized to a shape of 66 height and 200 width which was done in the Nvidia model
- the color scale was transformed to the YUV scale which separates brightness from color. This was also done in the Nvidia model
- Finally, the image pixel values were scaled to be between -0.5 and 0.5 which made stocahstic gradient descent much faster

### 3. Generator

A python generator was created that generate images in batches and passed them to the keras generator "model.fit_generator()". All functions were now run through this generator which could processes without memory constraints.

### 4. Augmentation
Several augmetations done on images to generate additional training data and to simulate instances where a larger steering angle was needed.
- Vertical and horizontal shifts were applied to half of the images with low steering angles and a steering value of .02 per pixel was applied. These were to aid the vehicle in moving back to the center of the road when it moved close to a shoulder. 
- A rotation was applied to 20% of the center images with a zero steering angle. Half of the rotated images were also shifted. This to aid the vehicle in moving back to the center of the road if it wobbled.
- Half of the images and corresponding steering angles were then flipped producing double the amount of training data.

### 5. Parameters
The performance of the vehicle improved making it onto the bridge but not much further. Several changes to augmentation parameters, steering values, and data selection allowed the vehicle to travel just a little bit farther past the bridge.
- Only 50% of the of center images with no steering angles were used after some experimentation. The over representation of training images with a zero steering value was addressed and the vehicle turned more.
- Horizontal shifts resulted in a reduced change in steering values of 0.008 per pixel.
- The speed of the car was reduced to 0.1 to allow for more changes in steering
- Left and right images recieved a +0.25 and -0.25 to their steering values in order to move them back to the center of the road

### 6. Second model
After these steps, the vehicle travelled farther than it had ever but tended to wandered over the road and eventually go off of it. The [Nvidia CNN](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) was then substitued and after one training the vehicle drove much straighter and navigated the entire track.

The Nvidia CNN is much deeper with 9 layers. A normalization layer, five convolution layers and three fully connected layers. Normalization was done outside of the model for this simulator project. The model is trained to minimize the MSE and uses Relu's for activation. are present with smaller filters and strides than the previous mode. A major advantage of this model is that the authors write that it's configuration has been chosen through experimentation. It also ran quickly relative to the prior model.

Additional modifications such as the use of maxpooling and Elu's for activation, [which are most effective in models with more than five layers](https://arxiv.org/pdf/1511.07289v1.pdf), were not included. Such modifications may have slowed training and performance on the first track was adequate.

####Final model with number parameters and connections
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
conv1_1 (Convolution2D)          (None, 31, 98, 24)    1824        convolution2d_input_3[0][0]      
____________________________________________________________________________________________________
activation_10 (Activation)       (None, 31, 98, 24)    0           conv1_1[0][0]                    
____________________________________________________________________________________________________
conv2_1 (Convolution2D)          (None, 14, 47, 36)    21636       activation_10[0][0]              
____________________________________________________________________________________________________
activation_11 (Activation)       (None, 14, 47, 36)    0           conv2_1[0][0]                    
____________________________________________________________________________________________________
conv3_1 (Convolution2D)          (None, 5, 22, 48)     43248       activation_11[0][0]              
____________________________________________________________________________________________________
activation_12 (Activation)       (None, 5, 22, 48)     0           conv3_1[0][0]                    
____________________________________________________________________________________________________
conv4_1 (Convolution2D)          (None, 3, 20, 64)     27712       activation_12[0][0]              
____________________________________________________________________________________________________
activation_13 (Activation)       (None, 3, 20, 64)     0           conv4_1[0][0]                    
____________________________________________________________________________________________________
conv4_2 (Convolution2D)          (None, 1, 18, 64)     36928       activation_13[0][0]              
____________________________________________________________________________________________________
activation_14 (Activation)       (None, 1, 18, 64)     0           conv4_2[0][0]                    
____________________________________________________________________________________________________
flatten_3 (Flatten)              (None, 1152)          0           activation_14[0][0]              
____________________________________________________________________________________________________
dense_0 (Dense)                  (None, 1164)          1342092     flatten_3[0][0]                  
____________________________________________________________________________________________________
activation_15 (Activation)       (None, 1164)          0           dense_0[0][0]                    
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           116500      activation_15[0][0]              
____________________________________________________________________________________________________
activation_16 (Activation)       (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        activation_16[0][0]              
____________________________________________________________________________________________________
activation_17 (Activation)       (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         activation_17[0][0]              
____________________________________________________________________________________________________
activation_18 (Activation)       (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          activation_18[0][0]              
====================================================================================================
Total params: 1595511


### 7. Overfitting

Overfitting was addressed in a few ways. Dropouts of 0.25 were added after each layer. Early stopping was not used but the number of epochs was limited to five which appeared to show continuous reductions in loss. 

A validation set was not used it the models performance on the first track was the main measure of validity, a larger training was thought to be more beneficial, and dropouts had a similar effect.

However, the vehicle had a very poor performance on the second track. This could be explain by overfitting but it also could be explained as the training images being very different from the tunnel like road on the second track.  

## Next Steps

Here are some steps that could be taken to improve the model.:
- Collecting additional data
- More agumentations of light, shadows, road colors
- Filling or interpolating black/blank pixels on shifted images. These spaces could resemble different landscapes or features.

## Lessons learned

Building a CNN takes a lot of time. Having a orderly training strategy as early as possible would be greatly benefical. Additional data exploratory steps could also help determine what type of data a model will need. GPU processing is also a helpful. 