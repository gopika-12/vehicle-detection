# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goal of this project was to develop a computer vision system to detect vehicles found in dashcam footage. Udacity recommends a machine learning approach (SVM or similar) using color transforms or a Histogram of Oriented Gradients (HOG) feature extraction. However, I chose to use a deep learning approach, training a ConvNet on a data set and then applying a sliding window classification system to each frame of the video.

The steps of the project:

1. Develop and train a convolutional neural network using the provided vehicle / non-vehicle dataset
1. Using the trained model, analyze each frame of the dashcam video for cars using a sliding window search

Here's an example from the dataset I used to train:

Vehicle Image:

![vehicle]

Non-Vehicle Image:

![non-vehicle]

I used 8969 non-vehicle images, and 4356 vehicle images. The dataset imbalance was initially a concern of mine, and I considered duplicating car images to create a more even dataset. However, I'm fairly happy with my results, so I ended up not needing to balance the datasets.

## Training the neural net

The training portion of the project is contained in my `train.py` file.

I started with [NVIDIA's end to end vehicle control architecture](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), which I've had good results with previously. Using the pathlib library, I loaded the dataset:

```
car_imgs = list(Path('./data/vehicles').iterdir())
non_car_imgs = list(Path('./data/non-vehicles').iterdir())
```

Then I concatenated them together and created a train/validation split:

```
all_imgs = car_imgs + non_car_imgs # List of all filenames
random.shuffle(all_imgs)
train, val = train_test_split(all_imgs, test_size=0.2) # Split into a test/val dataset
```

The validation and train lists are at this point just lists of filenames, so I need to load the images into memory and create a label representing if there is a car in the image or not. To avoid loading tens of thousands of images into memory at once, I used a generator, where the images were lazily loaded and labels added based on where the image was coming from.

```
for filename in batch_filenames:
    if 'non' in filename: # It's a nonvehicle image
        label = 0
    else:
        label = 1
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Fixes cv2.imread
    images.append(image)
    labels.append(label)
```

Note: `cv2.imread` brings in the image as BGR, whereas the dashcam footage is going to be read in as RGB. This tripped me up for a bit -- make sure the training and test applications have the exact same style of data!

I also mirrored all the images for data augmentation, which increased the dataset size.

My model is very similar to NVIDIA's, with some dropout added between the convolution layers and the fully connected. I experimented with the number of epochs to train for, and found that 9-10 was a good range to stop at. Thanks to dropout, there was very little overfitting.

```
Epoch 9/9 │167/166 [================] - 8s 48ms/step - loss: 0.0142 - val_loss: 0.0080
```

The validation loss is almost half the training loss, but on previous epochs the two were pretty similar, so I decided to stop at 9.

Now that the model is trained, it's time to test it!

## Using the trained model

To analyze each video frame using the trained model, I used a Jupyter notebook, `Video.ipynb`. This allowed for easy data visualization and parameter tuning, as well as making the code easier to read and understand for a first time reader.

First, I loaded in an image from the training dataset and used the model to predict if it was a car or not:

```
filename = './data/vehicles/3.png'
image = cv2.imread(filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
model.predict(image[None, :, :, :], batch_size=1)
```

Unsurprisingly, it prints a car label: `0.98094404` -- very close to `1`, which was the label given to cars during the training process.

There are several helper functions involved in the sliding window search:

First, `slide_window()`, which creates a list of windows to search in the image based on the function parameters. This allowed me to easily change the boundaries of the sliding window search, the stride, and the size of the windows.

Second, the actual `search_windows()` function. This is where the pretrained model comes in -- it grabs each window, resizes it to be 64x64 (the image size the model was trained on), and gets a prediction. If the prediction is above the threshold (0.7 seemed to work well), then it flags that window to be drawn on the image.

### Smoothing and False Positive Detection

The labelling of the car in each frame is fairly jerky, and tends to jump around from frame to frame. Additionally, there are a few false positive present due to inaccuracies in the neural net. Both of these problems can be fixed by ensuring that all boxes labeled have been present for the past several frames. This was implemented using a heatmap: for each frame, the last 20 frames were analyzed and each pixel that was found to be part of a car was incremented, once per frame (thus, each pixel could have a maximum value of 20). All pixels that had a value of 10 or less were discarded, leaving only strong areas of interest that had a track record of being a car. With these pixels calulated, the `scipy.ndimage.measurements.label()` function was used to isolate these non zero pixels, and `cv2.rectangle()` was used to draw the bounding boxes. 

The main tuning needed was the boundaries / size of the sliding window search, and the threshold to set something a car. In order to reduce false positives, I isolated the sliding window search to the road area -- ignoring the sky and other areas that were not valuable to search. This also sped up computation, as there were fewer windows to analyze per frame of video.

## Video + Future Work

To see the final video produced using the neural network, please [click here](https://youtu.be/iCGRa37d61s).

While this classification pipeline performs well, there are several possible areas for future improvement. Firstly: a larger and more balanced dataset may increase classification accuracy of footage. Secondly, a more elegant sliding window approach is possible. Inferring data from previous frames to allow the system to leverage previous known areas of interest would probably increase accuracy and decrease computational load. Finally, the visualization could be cleaned up slightly by averaging all connected bounding boxes to create a polygon of some sort to represent the entire car.

As I train networks and investigating deep learning, I am continually surprised to see the wide variety of applications. It continues to be a very exciting area of research for me, and this was no exception. Tuning the model and finding small but major errors was a great time, and I really enjoyed watching this project progress from start to finish.


[//]: # (Image References)
[vehicle]: ./output_images/vehicle.png
[non-vehicle]: ./output_images/non-vehicle.png
