---
layout: post
title:  "Plant Disease Classification"
author: "Dhruv Makwana"
comments: true
---

<p>Crop diseases are a major threat to food security, but their rapid identification remains difficult in many parts of the world due to the lack of the necessary infrastructure. The combination of increasing global smartphone penetration and recent advances in computer vision made possible by deep learning has paved the way for disease diagnosis.</p>

<p>Every disease, pest, and deficiency leaves behind a specific pattern. Farmer's Helper recognizes these patterns.One photo is enough and you know what your plant is missing.</p>

Entire code of this can be found [here](https://github.com/DhruvMakwana/crop-disease-detection)

## Dataset
Here we have used plantvillage dataset. The PlantVillage dataset consists of 61,486 healthy and unhealthy leaf images divided into 39 categories by species and disease.

Dataset can be found [here](https://data.mendeley.com/datasets/tywbtsjrjv/1)

The classes uses in dataset are:
1. Apple_scab
2. Apple_black_rot
3. Apple_cedar_apple_rust
4. Apple_healthy
5. Background_without_leaves
6. Blueberry_healthy
7. Cherry_powdery_mildew
8. Cherry_healthy
9. Corn_gray_leaf_spot
10. Corn_common_rust
11. Corn_northern_leaf_blight
12. Corn_healthy
13. Grape_black_rot
14. Grape_black_measles
15. Grape_leaf_blight
16. Grape_healthy
17. Orange_haunglongbing
18. Peach_bacterial_spot
19. Peach_healthy
20. Pepper_bacterial_spot
21. Pepper_healthy
22. Potato_early_blight
23. Potato_healthy
24. Potato_late_blight
25. Raspberry_healthy
26. Soybean_healthy
27. Squash_powdery_mildew
28. Strawberry_healthy
29. Strawberry_leaf_scorch
30. Tomato_bacterial_spot
31. Tomato_early_blight
32. Tomato_healthy
33. Tomato_late_blight
34. Tomato_leaf_mold
35. Tomato_septoria_leaf_spot
36. Tomato_spider_mites_two-spotted_spider_mite
37. Tomato_target_spot
38. Tomato_mosaic_virus
39. Tomato_yellow_leaf_curl_virus

There are two versions of dataset one without augmentation and other with augmentation where augmentation is performed with 6 different techniques (flipping, Gamma correction, noise injection, PCA color augmentation, rotation, and Scaling).

Consider This file structure
-plantdisease
	-dataset
	-input
		-Apple_black_rot.jpg
		-Apple_cedar_apple_rust.jpg
		-Apple_healthy.jpg
		-Apple_scab.jpg
		-Background_without_leaves.jpg
		-Blueberry_healthy.jpg
	-models
		-rn.h5
	-src
		-dataset.py
		-train.py
		-predict.py
	-static
		-images
			-image_1.jpg
			-image_2.jpg
			-image_3.jpg
			-image_4.jpg
	-templates
		-index.html
		-result.html
	-upload
	-main.py
	-requirements.txt

## Prepare Dataset

First we need to downlaod dataset and place it under dataset folder. For downloading dataset run `src/dataset.py` file. let's see what it does

	# importing libraries
	import requests, zipfile, io

	"""
		Download images folder from given url, move it to dataset folder.
	"""

	# url for data without augmentation
	# url = "https://data.mendeley.com/datasets/tywbtsjrjv/1/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/Plant_leaf_diseases_dataset_without_augmentation.zip?dl=1"

	# url for data with augmentation
	url = "https://data.mendeley.com/datasets/tywbtsjrjv/1/files/b4e3a32f-c0bd-4060-81e9-6144231f2520/Plant_leaf_diseases_dataset_with_augmentation.zip?dl=1"
	response = requests.get(url)
	z = zipfile.ZipFile(io.BytesIO(response.content))
	z.extractall()

Here we are downloading zip file from given url unzipping it and placing it in dataset folder. Note that current code is used data with augmentation. 

run `python dataset.py` command to run this file 

## Train Model

Here we are going to use ResNet152V2 model to train for 10 epochs with early stopping. open train.py file and import following code

	# importing libraries
	from glob import glob
	from keras.models import Model
	from keras.preprocessing.image import ImageDataGenerator
	from keras.layers import GlobalAveragePooling2D
	from keras.layers.core import Dropout, Dense
	from keras.applications import ResNet152V2
	from keras.applications.resnet_v2 import preprocess_input
	from keras.optimizers import Adam
	import tensorflow as tf

Now setup directory path, and print number of total images,

	train_dir  = "../dataset/Plant_leave_diseases_dataset_with_augmentation"
	print("Number of Images are: {}".format(len(glob(train_dir + '/*/*'))))

Output of above code will be
	
	Number of images are: 61486

Now we need to perform some preprocessing for that we can use `keras.applications.resnet_v2.preprocess_input()` function. other than that we also need to split our data in training and testing and rescaling it. For splitting we have used 80-20 split for train and test. All of this things can be done within ImageDataGenerator class. Note that we are using data which is already augmentated so we are not performing any other kind of Augmentation in ImageDataGenerator class. We have used batch size of 32 you can change as per your configuration.

	# data augmentation
	aug = ImageDataGenerator(preprocessing_function = preprocess_input,
	validation_split = 0.20,
	rescale = 1./255)  

	batch_size = 16
	training_set = aug.flow_from_directory(train_dir,
		target_size = (224, 224),
		batch_size = batch_size,
		class_mode = "categorical",
		subset = "training")
	test_set = aug.flow_from_directory(train_dir,
		target_size = (224, 224),
		batch_size = batch_size,
		class_mode = "categorical",
		subset = "validation")


Now define architecture here we are importing prebuild ResNet152V2 with imagenet weights with input shape of (224, 224, 3). We are adding three layers GlobalAveragePooling2D(), Dropout() with dropout rate of 0.25 and Dense() with 39 nodes which is number of classes we have. Make trainable parameter to True as we are going to train whole model and print model summary.

	# define architecture
	baseModel = ResNet152V2(weights = "imagenet", include_top = False, input_shape = (224, 224, 3))
	headModel = baseModel.output
	headModel = GlobalAveragePooling2D()(headModel)
	headModel = Dropout(0.25)(headModel)
	headModel = Dense(39, activation='sigmoid', name = "resnet152v2_dense")(headModel)

	model = Model(inputs = baseModel.input, outputs = headModel, name = "ResNet152V2")

	model.trainable = True
	# print summary
	print(model.summary())

Now we are defining criteria to stop early training. We will stop training if validation accuracy got reached 98% after corresponding epoch completes. We are using Callback from tf.keras.callbacks to achieve this. Note that we are considering result of validation accuracy after epoch ends.

	class myCallback(tf.keras.callbacks.Callback):
		def on_epoch_end(self, epoch, logs={}):
			if(logs.get('val_accuracy') > 0.97):
				print("\nReached 97% accuracy so cancelling training!")
				self.model.stop_training = True

	callbacks = myCallback()

Compile the model using Adam optimizer with learning rate of 0.005 change this to see difference. We are using categorical_crossentropy loss and accuracy as our matrics.

	# compile model
	model.compile(optimizer = Adam(learning_rate = 0.005), loss = 'categorical_crossentropy', metrics=['accuracy'])

Start training for 10 epochs with trainig set and testing set. Use steps_per_epoch equal to training_set.samples//batch_size and validation steps to test_set.samples//batch_size. Don't forget to use callbacks we created for early stoping. 

	# start training
	H = model.fit_generator(training_set,
		steps_per_epoch = training_set.samples//batch_size,
		validation_data = test_set,
		epochs = 10,
		validation_steps = test_set.samples//batch_size,
		callbacks = [callbacks],
		verbose = 1) 

Result of above training is 

	Epoch 1/10
	3074/3074 [==============================] - 794s 258ms/step - loss: 0.7289 - accuracy: 0.7813 - val_loss: 0.5274 - val_accuracy: 0.8464
	Epoch 2/10
	3074/3074 [==============================] - 791s 257ms/step - loss: 0.2194 - accuracy: 0.9288 - val_loss: 0.2383 - val_accuracy: 0.9264
	Epoch 3/10
	3074/3074 [==============================] - 803s 261ms/step - loss: 0.1427 - accuracy: 0.9531 - val_loss: 0.1081 - val_accuracy: 0.9674
	Epoch 4/10
	3074/3074 [==============================] - 803s 261ms/step - loss: 0.1065 - accuracy: 0.9653 - val_loss: 0.1219 - val_accuracy: 0.9585
	Epoch 5/10
	3074/3074 [==============================] - 799s 260ms/step - loss: 0.0835 - accuracy: 0.9730 - val_loss: 0.1150 - val_accuracy: 0.9653
	Epoch 6/10
	3074/3074 [==============================] - ETA: 0s - loss: 0.0670 - accuracy: 0.9778
	Reached 97% accuracy so cancelling training!
	3074/3074 [==============================] - 793s 258ms/step - loss: 0.0670 - accuracy: 0.9778 - val_loss: 0.0773 - val_accuracy: 0.9769

Save the model in models directory after training completes. 
	
	# save the model to file
	model.save('../models/resnet152v2.h5')

run `python train.py` command to run this file.

## Test Model

Now lets test the model by uploading single picture and predicting its class. Open `predict.py` file and import following code
	
	import numpy as np
	import keras
	from keras.preprocessing.image import img_to_array
	from keras.models import load_model
	from keras.preprocessing import image
	import cv2 

Now to predict any image, we need image path so assign url of image to imagepath variable and load model which we trained in train.py file

	imagepath = "../input/Apple_black_rot.jpg"
	model = load_model("../models/rn.h5") 

Let's make dictionary to assign class with numbers in same order as used in training. This will help in printing original class value instead of number. 

	 output_dict = {'Apple_scab': 0,
					'Apple_black_rot': 1,
                    'Apple_cedar_apple_rust': 2,
                    'Apple_healthy': 3,
                    'Background_without_leaves': 4,
                    'Blueberry_healthy': 5,
                    'Cherry_powdery_mildew': 6,
                    'Cherry_healthy': 7,
                    'Corn_gray_leaf_spot': 8,
                    'Corn_common_rust': 9,
                    'Corn_northern_leaf_blight': 10,
                    'Corn_healthy': 11,
                    'Grape_black_rot': 12,
                    'Grape_black_measles': 13,
                    'Grape_leaf_blight': 14,
                    'Grape_healthy': 15,
                    'Orange_haunglongbing': 16,
                    'Peach_bacterial_spot': 17,
                    'Peach_healthy': 18,
                    'Pepper_bacterial_spot': 19,
                    'Pepper_healthy': 20,
                    'Potato_early_blight': 21,
                    'Potato_healthy': 22,
                    'Potato_late_blight': 23,
                    'Raspberry_healthy': 24,
                    'Soybean_healthy': 25,
                    'Squash_powdery_mildew': 26,
                    'Strawberry_healthy': 27,
                    'Strawberry_leaf_scorch': 28,
                    'Tomato_bacterial_spot': 29,
                    'Tomato_early_blight': 30,
                    'Tomato_healthy': 31,
                    'Tomato_late_blight': 32,
                    'Tomato_leaf_mold': 33,
                    'Tomato_septoria_leaf_spot': 34,
                    'Tomato_spider_mites_two-spotted_spider_mite': 35,
                    'Tomato_target_spot': 36,
                    'Tomato_mosaic_virus': 37,
                    'Tomato_yellow_leaf_curl_virus':38}
     output_list = list(output_dict.keys())

Now load image using opencv library and resize it to (224, 224) size. we also need to convert image to array and rescale it same as we did in training. 
	
	print("loading image")
	img = cv2.imread(imagepath)
	img = cv2.resize(img, (224,224))
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = img/255

Now pass this image to our model.predict method and print result.
	
	print("predicting output")
	prediction = model.predict(img)
	prediction_flatten = prediction.flatten()
	max_val_index = np.argmax(prediction_flatten)
	result = output_list[max_val_index]
	print(result)

To run this file execute `python predict.py` command. 