# APS360-FinalProject-Breed-Classification
# Introduction

The project focuses on developing an advanced dog breed identification
model using Convolutional Neural Networks (CNNs) to address key needs in
pet adoption, veterinary care, and regulatory compliance. The model
helps match pets with potential owners and supports animal welfare
organizations by accurately identifying dog breeds from images, which
enhances veterinary diagnostics and treatments. This project is
significant because the deep learning model offers a reliable solution
to improve the management and care of dogs in various contexts.

# Illustration

The figure below illustrates the overall model for dog breed
identification. The model leverages two key components: an SSD model and
an Alexnet model. The SSD model processes the raw image by detecting the
dog's position within it and then cropping the dog from the image to
create a focused input for the Alexnet model. Alexnet, the primary model
in this system, is composed of two main parts: a feature extractor and a
classifier. The feature extractor is responsible for identifying and
isolating key features from the input image, which are then passed to
the classifier. The classifier processes these features by connecting
all neurons to make a prediction about the dog breed. Finally, the
output layer of the model provides a probability distribution,
indicating the likelihood of the image corresponding to various dog
breeds, and identifies the most probable breed.


![The overall model of the dog breed
identification](https://github.com/user-attachments/assets/4a1652ff-2767-40b3-84cc-4291710de31b
)

# Background and Related Work 

## Work 1

A significant contribution in the field comes from the study \"Dog Breed
Identification Using Deep Learning\" as described by @ethan1. This
research employs the ResNet-50 model, a deep convolutional neural
network pre-trained on ImageNet and fine-tuned for the specific task of
dog breed identification using transfer learning. This approach has
shown testing accuracy of 87.53 percent in distinguishing among 120
different dog breeds across a dataset of 20,580 images. Their findings
underline the advantages of utilizing transfer learning to adapt
pre-trained models to specific fine-grained classification tasks.

## Work 2

In 2019, Djordje Batic and Dubravko Culibrk conducted a study to
identify individual dogs in photos on a social media platform. This
study has been discussed in detail at @christina2. Their study combined
transfer learning and object detection approaches on the Inception V3
and SSD Inception V2 machine learning architectures. Batic and Culibrk
took photos from, Pet2Net, a social platform used majorly for pet
owners, to create datasets for training and testing their model. As a
result, they achieved an accuracy of 94.59 percent in identifying
individual dogs in unconstrained images.

## Work 3

As documented in @christina3, a study by the Computer Engineering
Department of Kasetsart University explored the application of machine
learning techniques to enhance dog grooming services . Utilizing the
Stanford Dogs Dataset, the researchers trained a preliminary
Convolutional Neural Network (CNN) architecture to classify various dog
breeds. These breeds were then further categorized into four groups
based on each dog's hair length and size. Non-domestic dog breeds were
excluded from the training dataset, which consisted of 20,261 dog images
across 118 classes to improve the model's precision. The study focused
on the VGG16 and Inception V3 models to investigate the impact of
Transfer Learning on classification accuracy. The experimental results
demonstrated that employing different Convolutional Neural Network
algorithms significantly enhances the efficiency and accuracy of dog
breed classification.

## Work 4

The ECTI association from Thailand conducted a test to analyze how well
existing Convolutional Neural Networks (CNNs) machine learning models
identify various dog breeds using the Stanford Dogs Data set at @Tim1.
The model uses different data processing techniques such as data
augmentation and data processing to enhance the performance of the
testing result. One of the key outcomes of this experiment shows that
the highest testing accuracy was 78 percent after applying data
augmentation, compared to 75 percent before data augmentation. The
research demonstrates the efficiency of using CNN to handle multiclass
classification within the same domain, i.e. different dog species among
dogs.

## Work 5

A study from the School of Computing and Information Systems of the
University of Melbourne developed a mobile application for dog breed
detection and recognition based on deep learning as recorded in @lyla1.
This work uses a CNN-based method to detect dogs in complex images and
identify the breed of those dogs. Researchers collected five thousand
images of dogs with breed labels as the training dataset. Multiple
methods of processing the training images were used to enhance the image
qualities, ensuring maximum accuracy. The study used transfer learning
to allow for the convolutional neural layers to directly capture the
characteristics and details from the raw images. As a result, the
application achieved almost 85 percent accuracy for the 50 common dog
breeds and 64 percent accuracy for the 120 less common dog types.

# Data Processing

Our group used the Stanford Dogs Dataset to train and validate our
model. The dataset contains 20580 pictures of dogs from 120 different
breeds. Each breed has its own folder and all images within the folders
are labeled. The team first uploaded all images from the Stanford Dogs
Dataset to a shared Google Drive directory so that all members could
access the data. Afterward, three separate folders were created for
training, validation, and testing data in the shared directory. All
images from the dataset were shuffled and copied to the three newly
created folders with proportions of 70%, 15%, and 15%. The images were
then cropped to a size of 224\*224 to simplify the training process. 

A major problem our team encountered during the process of data
splitting was when members tried to visualize the images along with
their labels, it was found that the labels for dogs of certain breeds
were swapped. For instance, all photos of Samoyeds were labelled as
golden retrievers, which led to potential inaccuracy in our model. After
investigating the structure of DatasetFolder on Git Hub @christina5
@christina6, team members manually modified our datasets' classes and
class_to_index variables to solve this issue. Figure 2 visualizes some
sample images along with their labels from our team's final training
dataset.

![Pictures from the training set along with their
labels](https://github.com/user-attachments/assets/7bcecf19-6e8a-466e-896a-bfda8cd2c0b2
)

Another crucial step in the team's data refining process was the
implementation of the single shot detection method (SSD) @Tim2, which
can identify and separate the dog object from given images. This highly
enhances the model's accuracy by removing noise from the image. For
example, when a dog and a person are in the same frame, the image will
be cropped to the boundaries of the dog object before being sent to the
classification model. Figure 3 displays an image before and after using
the SSD.

![image](https://github.com/user-attachments/assets/0954d3b4-330d-456f-b3ca-ca66db346a5e)


Due to the extensive training time required for the full data set, the
team needed a more efficient approach. Members decided to shrink the
number of breeds used for training to 50 randomly selected breeds. For
each breed, 100 samples were chosen and divided into training, testing,
and validation sets with the same proportions of 70%, 15%, and 15% as
before. This adjustment greatly reduced training time, while maintaining
a diverse representation of dog breeds.

# Architecture

The architecture of the team's final model included helper functions
referenced from @Tim4. In order to perform dog breed classification,
Alexnet was chosen due to its effectiveness in image recognition tasks.
The modified Alexnet architecture begins with an input layer designed to
accept images of size 224 \[Length\] x 224 \[Height\] x 3 \[RGB\]. It
includes five convolutional layers, each has ReLU activation functions
to introduce non-linearity. The initial convolutional layer has kernel
size 11x11 with a stride of 4, followed by a max-pooling layer with a
kernel size of 3 and a stride of 2. The second convolutional layer has
kernel size 5x5, followed by a similar max-pooling layer. The remaining
three convolutional layers all have kernel size of 3x3. These
convolutional layers are designed to capture both low-level features
such as dog color, texture, and high-level features such as body size,
head position, etc. This information is important for identifying
different dog breeds.

After that, the model includes three fully connected layers, reducing
neuron sizes from 4096 to 512 to 256. Each fully connected layer is
followed by a ReLU activation function to introduce non-linearity. To
prevent overfitting, dropout layers with a dropout rate of 0.5 are also
applied. The final prediction is generated by a fully connected layer
with 50 output units, corresponding to the maxmine number of dog breeds
the model will learn.

# Baseline Mode

All team members collectively decided to build a convolution neural
network called simpleCNN to be our baseline model rather than the ANN
model as initially proposed on the progress report. SimpleCNN consists
of two convolutional layers as well as two max pooling layers. The max
pooling layers prevent overfitting and help speed up the program. Two
fully connected layers were also used, the first layer projects the
vector to 128 classes, and the final layer predicts the result over the
50 number of dog breeds. This structure of simpleCNN is visualized in
Figure 4 below.

![image](https://github.com/user-attachments/assets/09ed6b78-ba47-4fb7-b912-4d00b527636a)


The dataset used for the baseline model is picked from the Stanford Dogs
Dataset as well. Instead of using all images, team members picked only
photos of 5 different breeds of dogs for the purpose of training and
testing simpleCNN. The images have been processed as explained in
section 4.0, except SSD was not applied. Using functions from previous
labs in this course and cross-entropy loss as the loss function,
simpleCNN is found to have an accuracy of only 0.2.

Later on, our team tried increasing the number of convolutional layers
in the hope of increasing the accuracy of the baseline model. However,
Google Colab crashed due to insufficient amount of RAM. The team then
decided to use just simpleCNN.

# Quantitative Results

To evaluate the performance of our model, several quantitative
measurements were conducted. Initially, the test accuracy of the model
was recorded as 40.71 percent before any hyperparameter tuning. Through
continuous adjustments, the accuracy was significantly improved to 78.77
percent, demonstrating the effectiveness of the tuning process. After
that, the implementation of the Single shot detection (SSD) method
further enhanced the model's performance by raising test accuracy to
85.69 percent.

Important stages of the model's training curves(Figure 5) and
losses(Figure 6) are recorded below, providing insights on the model's
performance as it improves.

![image](https://github.com/user-attachments/assets/650aaf46-4fd5-422b-8bbe-86460842c3c9)

![image](https://github.com/user-attachments/assets/24e5626b-f738-4e5b-8355-0cc83d25fe1c)


# Qualitative Results

To evaluate the performance of our dog breed identification model, we
randomly selected several images from the testing folder. As illustrated
in Figure 7, the test set includes two Gordon Setters, three Chihuahuas,
and four Mexican Hairless dogs. The model correctly identified seven out
of the nine dogs(Figure 8), resulting in an accuracy rate of
approximately 78 percent. This outcome indicates that the model is
fairly effective at classifying different dog breeds, though there is
still room for improvement. The misclassifications highlight areas where
the model might need further refinement, particularly in distinguishing
between breeds with similar features. Overall, the results suggest that
the model performs well, but continued optimization could enhance its
accuracy.

![image](https://github.com/user-attachments/assets/cf55b12c-93d3-4095-bfe6-9049afabed7f)

In the case of the SSD model, the input is displayed in Figure 9. After
processing the data with the SSD model, the dog's position in the raw
images is accurately detected. The outputs, as shown in Figure 10,
demonstrate that the SSD model successfully identified and cropped the
dog's position in each image. These accurately cropped images are then
used as input for the AlexNet model. The SSD model effectively isolates
the dog from both the background and any people present, ensuring that
the AlexNet model receives focused and relevant input, which is crucial
for accurate breed identification. This process underscores the SSD
model's efficiency in detecting and preparing image data for the AlexNet
model, significantly contributing to the overall success of the dog
breed identification system.

![image](https://github.com/user-attachments/assets/65b5ffdd-bd8c-47fd-900b-c0dd27c17f9b)


# Evaluate model on new data

To ensure the model's performance is accurately represented on new data,
we employed two evaluation methods. First, we tested the model with dog
images we took ourselves(Figure 11). Among those images, only one image
where person and dog are in the same frame resulted in a misprediction.
The overall accuracy was 83.3 percent. This phenomenon led to the
implementation of the single shot detection method (SSD) described in
section 4.0, which greatly optimized this problem.

![image](https://github.com/user-attachments/assets/1c200762-4c42-4828-909d-76b1ec1d53df)


The second method was to test the model on an entirely new dataset. As
the model was initially trained using the Stanford dogs dataset
containing 120 dog breeds and 20,580 images, the TsingHua dogs dataset
@Tim3 (Figure 12) with 130 dog breeds and more than 70000 images is now
introduced. This dataset was completely new to the model, providing a
thorough test to its abilities. Due to dataset structure differences, a
small portion where both data sets have in common was selected for
testing. With SSD applied, the model achieved a testing accuracy of 75.5
percent on the Tsinghua dataset, compared to 85.69 percent on the
Stanford dataset. Although there was a slight decrease in testing
accuracy, the model's predictions on new data was still relatively high.
This indicates the model's excellent performance on solving the dog
breed classification problem.

![image](https://github.com/user-attachments/assets/6f777b93-ff9b-4260-b19b-a64c8d82451b)

# Discussion

The model demonstrates high performance with a testing accuracy of 85.69
percent. Moreover, the model's performance on unseen data reached a high
accuracy of 75.5 percent. Not only does this accuracy surpass our team's
baseline model, simpleCNN, by 65.59 percent, it also exceeds the 78
percent testing accuracy reported by @Tim1 in one of the studies
mentioned previously in section 3.4. This result indicates that the
team's modifications on the AlexNet architecture, along with the data
processing methods significantly improved a model's ability to
distinguish between different dog breeds.

An interesting observation is that after our team used the SSD to
preprocess the images by cropping out the humans in them, the test
accuracy on unseen data didn't increase, but decreased from 78 percent
to 75.5 percent. The team originally expected the model to perform
better as we thought the edited pictures would prevent the network from
focusing on irrelevant features. Members analyzed that the potential
reasoning behind this phenomenon would be because our network has never
seen photos with both dogs and humans in them while being trained, and
has mistakenly identified some human features as key features in
differentiating dog breeds. On top of that, the testing dataset for
cropped images is a lot smaller than the testing dataset for uncropped
images. This issue might have led to imprecise measurements in the
correctness of our model's predictions, giving rise to unfair
comparison. To prevent this situation from happening in the future, a
cropping feature could be incorporated into the network so that when new
images get passed to the model, they would be cropped before being
analyzed. The team will also use the same number of images for testing
various networks.

Overall, the results prove that the model is performing well and
validates the network's performance. Future work could focus on further
optimizing the model to handle cases with challenging backgrounds with
different kinds of animals in them. The model could also be improved to
receive images with multiple dogs in it and classify all of them.

# Ethical Considerations

Although our team has been making a concerted effort to avoid giving
rise to ethical problems when choosing the main focus of our project,
our chosen theme of creating a model to identify different dog breeds
could lead to potential ethical issues. One of the most prominent
ethical concerns is the potential of our model leading to breed
discrimination and overemphasis on breed in dogs. Since some breeds are
subject to certain stereotypical breed features, misidentification of
these dogs could lead to potential unjust treatment of dogs. In
addition, our design's misidentifications might increase people's
obsession with pedigree dogs over mixed breed dogs. This obsession could
lead to animal breeding, which in some cases violates the animal
integrity of the dogs being bred, and creates genetic problems in some
dog breeds as described in @christina4. On top of that, overly focusing
on dogs' breeds could lead to neglecting individual dog's behavioural
needs. Dog owners might treat their dogs according to a certain breed's
stereotypical behavioural features rather than observing their own dog's
special personality. Another possible ethical problem arises during data
collection for training our model. Our team needs to ensure that all
photos used for training are ethically sourced and have consent from the
dog owners.

# References
![image](https://github.com/user-attachments/assets/0c9dbed0-529d-468a-878b-4de92ba48db3)
![image](https://github.com/user-attachments/assets/e5e6cd8d-bc52-4955-93ec-f5dd3765afcf)

