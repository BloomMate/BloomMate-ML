# BloomMate - Machine Learning

> This software is part of a project for a software engineering class at **Hanyang University Information Systems Dept.**

> The class is in collaboration with **LG Electronics.**

## ① Intro & Motivation

When designing BloomMate, our team centered around "whether or not the plant's owner is in the SmartCottage." For each situation, the core functionality is as follows

1. The owner is not in the SmartCottage
   - use generative AI to talk to the plant.
2. The owner is in the SmartCottage
   - take a picture of the plant and **use AI to diagnose it.**

This Machine Learning Repository is for the AI used in the second situation. Train the AI using Custom Dataset and convert it to tensorflow lite so that it can be used in real projects.

## ② Datasets

As we'll see later, Resnet-50 is a model that has been pre-trained on multiple datasets. However, BloomMate needs to diagnose plant diseases for four different crops (corn, potatoes, tomatoes, and strawberries). Unfortunately, Resnet-50's model for image classification was not pre-trained with the right data for this situation.

Therefore, we had to find a custom dataset. Fortunately, we found the best one for our situation at this [link](https://data.mendeley.com/datasets/tywbtsjrjv/1). However, we had a problem with over-fitting in the first training, so we set the data for all classes to 1000.

#### Dataset Classes

| 🍅 tomato | 🥔 potato | 🌽 corn | 🍓 strawberry |
| --------- | --------- | ------- | ------------- |
| to        | po        | co      | stra          |

## ③ Model (Resnet-50)

At first, we followed [tensorflow's image classification guide](https://www.tensorflow.org/tutorials/images/classification), which was great for distinguishing between completely different shapes of leaves (e.g., tomatoes and corn), but once we started to have more classes that needed to be distinguished, such as diseased or not diseased, it didn't work as well. So we started looking for other proven models, and chose Resnet-50.

ResNet-50 is a convolutional neural network (CNN) architecture commonly used for building deep neural networks. It is part of the ResNet (Residual Network) series developed by Microsoft Research, known for its outstanding performance in image recognition and classification tasks.

Key features of ResNet-50:

1. Depth: ResNet-50 consists of 50 layers, representing a deep neural network. This depth addresses the vanishing gradient problem during training and allows for high-level abstraction.

2. Residual connections: ResNet introduces residual connections, enabling each block to learn the residual (difference) between input and output. This facilitates the flow of information without loss, easing training and improving performance.

3. Transfer learning: Pre-trained on the large-scale ImageNet dataset, ResNet-50 serves as a pre-trained model. This allows for transfer learning, applying the knowledge gained from ImageNet to various computer vision tasks.

4. Batch normalization: ResNet incorporates batch normalization at each layer to stabilize training and accelerate learning.

Resnet-50 models deployed in tensorflow or pytorch are pre-trained. However, as mentioned above, the images we wanted were not pre-trained and we had to train the model on our own dataset and fine-tune it. The main variables and parameters used for training are as follows.

```python
### Image Dimension
height,width=180,180

### Fine Tuning Code
dnn_model.add(imported_model)
dnn_model.add(Flatten())
dnn_model.add(Dense(512, activation='relu'))
dnn_model.add(Dense(5, activation='softmax'))

### Compile Deep learning
dnn_model.compile(
    optimizer="adam",
    loss='categorical_crossentropy',
    metrics=['accuracy'])

### Training Cycle
epochs=10
```

The results of the training are shown below, a graph of loss versus accuracy, which looks somewhat idealized. (The details will vary with each training.)

![Alt text]('./training-result.png')

If you take a look at the code result, you'll see that it says that the corn is not diseased with 100% accuracy for corn that is not normally diseased. (It was harder to get leaves from diseased/non-diseased plants than I thought, because people are all interested in the fruit rather than the leaves 😅)

> 1/1 [==============================] - 0s 93ms/step  
> This image most likely belongs to Corn\_\_\_healthy with a 100.00 percent confidence.

## ④ Conversion Into Tensorflow lite

We then needed to apply our trained model to the BloomMate service. We had a number of options, but we chose to convert to tensorflow-lite and run the model on the backend. Here's why.

1. tensorflow-lite reduces the size of the model (performance remains largely unchanged), so the smaller model is less likely to overload the backend server.
2. The frontend cannot maintain a bundle size of more than 100MB due to the policies of the Play Store and App Store. The BloomMate application is about 60MB, and even though tensorflow-lite's model is lightweight, it is close to 90MB, so it was not possible to port the model and make it work on the frontend.

So we decided to run the model on a backend powered by Django. (We designed the software from the ground up so that the programming language was the same.) The code for that can be found at [this link](https://github.com/BloomMate/BloomMate-BE/blob/main/plants/utils.py#L45). Finally, check out the video and screenshots below to see the diagnosis in action in BloomMate!

#### Screenshots

|`Strawberry-healthy`|`Strawberry-Leaf-Scortch`|
|--------------------|-------------------------|
|<img src="https://github.com/BloomMate/BloomMate-ML/assets/60422588/d7f33f5f-b785-4679-aaed-d66b526d18b0" width="275" height="550"/>|<img src="https://github.com/BloomMate/BloomMate-ML/assets/60422588/a9955a08-ba88-48f3-bf19-c9d36a9df324" width="275" height="550"/>|

## Reference

- https://www.tensorflow.org/tutorials/images/classification
- https://data.mendeley.com/datasets/tywbtsjrjv/1
- https://medium.com/@bravinwasike18/building-a-deep-learning-model-with-keras-and-resnet-50-9dd6f4eb3351
