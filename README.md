# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goal of this project was to develop a computer vision system to detect vehicles found in dashcam footage. Udacity recommends a machine learning approach (SVM or similar) using color transforms or a Histogram of Oriented Gradients (HOG) feature extraction. However, I chose to use a deep learning approach, training a ConvNet on a data set and then applying a sliding window classification system to each frame of the video.

The steps of the project:

1. Develop and train a convolutional neural network using the provided vehicle / non-vehicle dataset
1. Using the trained model, analyze each frame of the dashcam video for cars using a sliding window search

## Training the neural net

To train the model, I used `train.py`.

I started with NVIDIA's end to end vehicle control architecture, which I've had good results with previously. Using the pathlib library, I loaded the dataset:

```
car_imgs = list(Path('./data/vehicles').iterdir())
non_car_imgs = list(Path('./data/non-vehicles').iterdir())
```

Here's a sample vehicle and non-vehicle image:

![vehicle]
![non-vehicle]

Then I concatenated them together and created a test/train split:

```
all_imgs = car_imgs + non_car_imgs # List of all filenames
random.shuffle(all_imgs)
train, val = train_test_split(all_imgs, test_size=0.2) # Split into a test/val dataset
```

This long list of filenames was then passed into my generator function, where the images were lazily loaded and labels added based on where the image was coming from.

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

My model is very similar to NVIDIA's, with some dropout added between the convolution layers and the fully connected.

```
Epoch 9/9 │167/166 [================] - 8s 48ms/step - loss: 0.0142 - val_loss: 0.0080
```

After nine epochs, things were looking good.  The validation loss is almost half the training loss, but some epochs it was pretty similar, so I decided to stop at 9.

Now that the model is trained, it's time to test it!

## Using the trained model

To use the trained model, I used `Video.ipynb`. First, I loaded in an image from the training dataset and used the model to predict if it was a car or not:

```
filename = './data/vehicles/3.png'
image = cv2.imread(filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
model.predict(image[None, :, :, :], batch_size=1)
```

Unsurprisingly, it prints a car label: ` `

There are several helper functions involved in this process:

First, `slide_window()`, which creates a list of windows to search in the image based on the function parameters. This allowed me to easily change the boundaries of the sliding window search, the stride, and the size of the windows.

Second, the actual `search_windows()` function. This is where the pretrained model comes in -- it grabs each window, resizes it to be 64x64 (the image size the model was trained on), and gets a prediction. If the prediction is above the threshold (0.8 seemed to work well), then it flags that window to be drawn on the image.

Finally, the `draw_boxes()` function, which takes all the flagged windows and calls `cv2.rectangle()`.

The entire process is fairly straightforward. The main tuning needed was the boundaries / size of the sliding window search, and the threshold to set something a car.

## Summary

As I continue training networks and investigating deep learning, I continue to see the wide variety of impressive applications. It continues to be a very exciting area of research for me, and this was no exception. Tuning the model and finding small but major errors was a great time, and I really enjoyed watching this project progress from start to finish.


[//]: # (Image References)
[vehicle]: ./data/vehicles/3.png
[non-vehicle]: ./data/non-vehicles/image986.png
