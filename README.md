# SVM_image_classification

The fundamental idea of classifying images has lead to
revolutionary developments in the field of Computer Vision.

Though the current applications have been extended to Object detection,
Object tracking, Image generation, Image enhancements, Video Analysis and
so on.., the success of image classification techniques stands as a milestone
for the progress in the above mentioned technologies.

The core of all these training algorithms includes the convolutional 
layers, which enables the model to understand the features of an image.
But, was it possible to perform image classification even before the dawn
of Neural Networks?? Ofcourse yes, and one of the main contributor for
that is a popular Machine Learning Algorithm 'Support Vector Machine (SVM)'.

Idea behind SVM image Classifier:

1. Each image(2D) is converted to a feature vector(1D) and labelled.
2. The model trains on all these feature vectors to find an optimal
hyperplane.
3. This hyperplane distinguishes between feature vectors of different labels.


How to create a feature vector?
<ul>
  <li>Read an image of any size : ex., (200,300,3)</li>
  <li>Resize to same shape (200,200,3), which means width=200 px, height=200 px, channels = 3 </li>
  <li> Flatten the image (200*200*3), the vector shape is (120000,) </li>
 </ul>
This is the feature vector of an image.

<h3> Test Results:</h3>

![s1](https://github.com/hilary-anish/SVM_image_classification/assets/110568431/58dabdab-ab14-4f39-b1b5-21fa421a35ba)
![f3](https://github.com/hilary-anish/SVM_image_classification/assets/110568431/445a2e4b-8760-4732-9fba-9a47190f622e)
![m2](https://github.com/hilary-anish/SVM_image_classification/assets/110568431/cf5db27d-afc0-4a82-b0c7-2076fdc21afa)



The dataset is available in Kaggle [Intel_dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
