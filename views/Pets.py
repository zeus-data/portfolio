import streamlit as st

# st.write('''#### Work in Progress . . . ''')

# st.image('images/atlantis.png')

st.write('''## Convolutional Neural Networks and an Algorithm for a 🐕 Identification App
( and a 👩👨‍🦰 face detector ) ''')

st.image('./images/dog_breed/face_detector.png') 


st.write(''' In this project, I will build a pipeline to process real-world, user-supplied images.
          Given an image of a dog, the algorithm will identify an estimate of the canine’s breed.
          If supplied an image of a human, the code will identify the resembling dog breed.
         
In this notebook, I will make the first steps towards developing
          an algorithm that could be used as part of a mobile or web app.
          At the end of this project, the code will accept any user-supplied image as input.
          If a dog is detected in the image, it will provide an estimate of the dog's breed.
          If a human is detected, it will provide an estimate of the dog breed that
          is most resembling.
          The image below displays potential sample output of the finished project''')



left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('images/sample_dog_output.png', width=500)


st.write('''### Step 0: Import Datasets

##### Import Dog Dataset

In the code cell below, I load a dataset containing images of dogs. The dataset is
          processed using the load_files function from the scikit-learn library. 
         This function helps organize and retrieve the data efficiently. A few variables
          are initialized and populated during this process, making the dataset ready for analysis. 
         This setup lays the groundwork for working with the dog image data in subsequent steps.
- `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
- `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels 
- `dog_names` - list of string-valued dog breed names for translating labels''')


code='''
from sklearn.datasets import load_files       
from tensorflow.keras.utils import to_categorical
import numpy as np
from glob import glob

%matplotlib inline 
'''
st.code(code)

from sklearn.datasets import load_files       
# from keras.utils import np_utils
import numpy as np
from glob import glob

code='''# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test and validation datasets 
train_files, train_targets = load_dataset('./images/dog_breed/dogimages/train')
valid_files, valid_targets = load_dataset('./images/dog_breed/dogimages/valid')
test_files, test_targets = load_dataset('./images/dog_breed/dogimages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("./images/dog_breed/dogimages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
'''

st.code(code, line_numbers=True) 

st.write('''There are 133 total dog categories.
         
There are 8351 total dog images.

There are 6680 training dog images.
         
There are 835 validation dog images.
         
There are 836 test dog images.
         
         ''')


def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets
 



# load train, test and validation datasets 
train_files, train_targets = load_dataset('./images/dog_breed/dogimages/train')
valid_files, valid_targets = load_dataset('./images/dog_breed/dogimages/valid')
test_files, test_targets = load_dataset('./images/dog_breed/dogimages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("./images/dog_breed/dogimages/train/*/"))]

# print statistics about the dataset
# st.write('There are %d total dog categories.' % len(dog_names))
# st.write('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
# st.write('There are %d training dog images.' % len(train_files))
# st.write('There are %d validation dog images.' % len(valid_files))
# st.write('There are %d test dog images.'% len(test_files)) 


st.write('''##### Import Human Dataset
In the code cell below, I load a dataset containing images of humans.
          The file paths for these images are stored in a NumPy array named `human_files`. 
         This array organizes the dataset, allowing easy access to individual image files.
          It serves as the foundation for processing and analyzing the human images.''')


code='''import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("./images/dog_breed/humanimages/lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print(f' The dataset contains a total of  {len(human_files):,} human images. Below is an example of one of these images.')'''
st.code(code)

st.write(''' The dataset contains a total of 13,233 human images. Below is an example of one of these images.''')


import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("./images/dog_breed/humanimages/lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
# st.write(f'There are {len(human_files):,} total human images.')



code='''import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread(human_files[16])
imgplot = plt.imshow(img)
'''
st.code(code)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread(human_files[5])
st.image(img, caption="Silvester Stalone", width=300)

# Optionally, if you want to use matplotlib for additional customization
fig, ax = plt.subplots()
ax.imshow(img)



st.markdown('''### Step 1: Detect Humans
I will be using `OpenCV`'s implementation of Haar feature-based cascade classifiers to detect human faces in images.
             OpenCV provides many pre-trained face detectors, stored as XML files on github. 
            I have downloaded one of these detectors to use.

How the code works: Before using any of the face detectors, it is standard procedure to convert the images
             to grayscale. The `detectMultiScale` function executes the classifier stored in face_cascade and takes
             the grayscale image as a parameter.

In the code below, faces is a numpy array of detected faces, where each row corresponds to a detected face.
             Each detected face is a 1D array with four entries that specifies the bounding box of the detected
             face. The first two entries in the array (extracted in the above code as x and y) specify the
             horizontal and vertical positions of the top left corner of the bounding box.
             The last two entries in the array (extracted here as w and h) specify the width and height of
             the box.
''')

code='''import cv2
import numpy as np
from PIL import Image

# Load the pre-trained face detector 
face_cascade = cv2.CascadeClassifier('images/dog_breed/haarcascade_frontalface_alt.xml')

# Upload an image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Convert BGR image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray)

    # Print number of faces detected
    st.write(f"Number of faces detected: {len(faces)}")

    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert BGR image to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image with bounding boxes
    st.image(img_rgb, caption="Detected Faces")'''
st.code(code)


import cv2
import numpy as np
from PIL import Image

# Load the pre-trained face detector (ensure the haarcascade XML file is in the correct path)
face_cascade = cv2.CascadeClassifier('images/dog_breed/haarcascade_frontalface_alt.xml')

# Upload an image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Convert BGR image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray)

    # Print number of faces detected
    st.write(f"Number of faces detected: {len(faces)}")

    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert BGR image to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image with bounding boxes
    st.image(img_rgb, caption="Detected Faces")

st.write('''**Would you like to give it a try**? 
            Upload an image containing a person's face (or multiple faces) and see if 
            the model successfully detects them!''')

st.write(''' #### Step 2: Detect Dogs
In this section, I will use a pre-trained ResNet-50 model to detect dogs in images.
         This code downloads the ResNet-50 model, along with weights that have been trained
          on ImageNet, a very large, very popular dataset used for image classification and other vision tasks.
          ImageNet contains over 10 million URLs, each linking to an image containing an object from one of
          1000 categories. Given an image, this pre-trained ResNet-50 model returns a prediction
          (derived from the available categories in ImageNet) for the object that is contained in the image.''')


code='''from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')'''
st.code(code)

st.image('images/dog_breed/keras.png')


st.write('''##### Pre-process the Data

When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor)
          as input, with shape (**nb_samples**, **rows**, **columns**, **channels**), where `nb_samples`
          corresponds to the total number of images (or samples),
          and `rows`, `columns`, and `channels` correspond to the number of rows,
          columns, and channels for each image, respectively.  

The `path_to_tensor` function below takes a string-valued file path to a color image as input and
          returns a 4D tensor suitable for supplying to a Keras CNN.  The function first loads the
          image and resizes it to a square image that is **224 times 224** pixels.  Next, the image is
          converted to an array, which is then resized to a 4D tensor.  In this case, since we are working
          with color images, each image has three channels.  Likewise, since we are processing a single image
          (or sample), the returned tensor will always have shape (**1, 224, 224, 3**).

The `paths_to_tensor` function takes a numpy array of string-valued image paths as input and 
         returns a 4D tensor with shape (`nb_samples`, 224, 224, 3).

Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths. 
          It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds
          to a different image) in the dataset!''')


code='''from keras.preprocessing import image                  
from tqdm import tqdm


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)'''
st.code(code)


st.write('''#### Making Predictions with ResNet-50

Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras,
          requires some additional processing.  First, the RGB image is converted to BGR
          by reordering the channels.
           All pre-trained models have the additional normalization step that the mean pixel
          (expressed in RGB as [103.939, 116.779, 123.68] and calculated from all pixels in
          all images in ImageNet) must be subtracted from every pixel in each image. 
          This is implemented in the imported function `preprocess_input`. 
         
Now that there is a way to format the image for supplying to ResNet-50,
          the model is ready to extract the predictions. 
          This is accomplished with the `predict` method, which returns 
         an array whose $i$-th entry is the model's predicted probability
          that the image belongs to the $i$-th ImageNet category.
           This is implemented in the `ResNet50_predict_labels` function.

By taking the argmax of the predicted probability vector, the result is an integer
          corresponding to the model's predicted object class, which can be identified
          with an object category through the use of this
          [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). ''')  

code='''from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))'''
st.code(code)

st.write('''#### Write a Dog Detector

In the dictionary, the categories corresponding to dogs appear in an uninterrupted
          sequence and correspond to dictionary keys 151-268, inclusive, to include all
          categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check
          to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model,
          the `ResNet50_predict_labels` function should return a value
          between 151 and 268 (inclusive).

The `dog_detector` function below, returns
          `True` if a dog is detected in an image (and `False` if not).''')

code='''def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) '''
st.code(code)


st.write('''#### Step 3: Build a CNN to Classify Dog Breeds (from scratch)

After implementing functions to detect humans and dogs in images
         , the next step involves predicting the breed from these images. This requires building a Convolutional Neural Network (CNN) specifically designed to classify dog breeds. Initially, I will create a CNN from scratch, without transfer learning, and aim to achieve a minimum test accuracy of 1%. In subsequent steps, transfer learning will be applied to develop a more advanced
          CNN with significantly higher accuracy.

It’s important to recognize that adding multiple trainable 
         layers to the CNN increases the number of parameters,
          which can significantly extend training time. As a result,
          using a GPU is essential for speeding up the process. Fortunately,
          Keras provides an estimate of how long each epoch will take during training.
          Furthermore, identifying dog breeds from images is an inherently complex task. 
         Even humans often find it difficult to differentiate between similar breeds, 
         such as the Brittany and the Welsh Springer Spaniel.''')

st.write('''#### Pre-process the Data

I will rescale the images by dividing every pixel in every image by 255.''')

code='''from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files[:4000]).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255'''
st.code(code)


st.image('images/dog_breed/CNN_training.png')


st.write('''##### My model architecture is as follows:

- The CNN model has three convolutional layers and the network is capable of learning
          increasingly complex and abstract features as it progresses through these layers.
          These layers are responsible for extracting local features from the input images.
          They use filters to capture patterns, edges, and textures.

- A max pooling layer follows each convolutional layer. This layer reduces the spatial
          dimensions of the learned features by retaining the maximum value in each local
          region. This downsampling helps to decrease the computational load.

- Following the layers above, the global average pooling layer computes the average value
          of each feature map across its entire spatial dimensions. This results in a single
          value for each feature map, effectively summarizing the presence of each learned
          feature.

- The dense layer at the end connects every neuron from the previous layer to every neuron
          in its layer. This layer aggregates and combines these learned features from different
          spatial locations. The output of the dense layer is then fed into a softmax activation
          function, providing probability scores for different classes.



Convolutional Neural Networks are useful for image classification due to their specialized 
         architecture that takes into account the spatial relationships and hierarchical features
          present in images.  CNNs use convolutional layers to scan small area of an image at
          a time. This allows them to capture local features such as edges and patterns. 

They consist of multiple layers, each responsible for learning different levels of abstraction.
          This enables to understand and recognize intricate patterns in the data. Also, CNNs
          benefit from their pooling layer ability; they retain the most important information 
         in a local region with the ultimate goal of improved generalization. Finally, CNNs use
          backpropagation and gradient descent during training to update the
          weights and biases, optimizing the network for accurate classification. ''')

code='''from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential'''
st.code(code)

code='''model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu',
input_shape=train_tensors.shape[1:]))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(133, activation='softmax'))

model.summary()'''
st.code(code)


st.image('./images/dog_breed/model_architecture.png')


st.write('''##### Compile the Model''')

code='''model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
 metrics=['accuracy'])'''
st.code(code)


st.write('''##### Train the Model ''')

code='''from keras.callbacks import ModelCheckpoint
epochs = 20

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets[:4000], 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)  '''
st.code(code)

st.image('''images/dog_breed/model_training.png''')



st.write('''##### Load the Model with the Best Validation Loss''')

code='''model.load_weights('saved_models/weights.best.from_scratch.hdf5')'''
st.code(code)


st.write('''##### Test the Model

Try out your model on the test dataset of dog images.
           Ensure that your test accuracy is greater than 1%.''')


code='''# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.2f%%' % test_accuracy)'''
st.code(code)


st.write('''Test accuracy: 3.11%''')


st.write('''---
### Step 4: Use a CNN to Classify Dog Breeds

To reduce training time without sacrificing accuracy, we show you how to train a CNN
          using transfer learning.  In the following step, you will get a chance to use
          transfer learning to train your own CNN.

#### Obtain Bottleneck Features''')

code='''bottleneck_features = np.load('/data/bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']'''
st.code(code)


st.write('''### Model Architecture

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last
          convolutional output of VGG-16 is fed as input to our model.  We only add a global
          average pooling layer and a fully connected layer, where the latter contains
          one node for each dog category and is equipped with a softmax.''')

code='''VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()'''
st.code(code)

st.image('''images/dog_breed/model_training_2.png''')


st.write('''#### Load the Model with the Best Validation Loss''')

code='''VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')'''
st.code(code)

st.write('''#### Test the Model

Now, we can use the CNN to test how well it identifies breed within our test dataset of
          dog images.  We print the test accuracy below.''')

code='''# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0)))
 for feature in test_VGG16]

# report test accuracy
test_accuracy = np.sum(np.array(VGG16_predictions)==np.argmax
(test_targets, axis=1))/len(VGG16_predictions)
print(f'test accuracy: {test_accuracy:.2%}')'''
st.code(code)

st.write('''test accuracy: 37.1%''')


st.write('''#### Predict Dog Breed with the Model''')

code='''from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]'''
st.code(code)


st.write('''---
#### Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

I will now use transfer learning to create a CNN that can identify dog breed from images.
           The new CNN must attain at least 60% accuracy on the test set.

In Step 4, I used transfer learning to create a CNN using VGG-16 bottleneck features.
           In this section, I will use the bottleneck
          features from a different pre-trained model. 
          To make things easier for you,
          we have pre-computed the features for all of the networks
          that are currently available in Keras.
           These are already in the workspace, at /data/bottleneck_features. 
          If you wish to download them on a different machine, they can be found at:
- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

The files are encoded as such:

    Dog{network}Data.npz
    
where `{network}`, in the above filename, can be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.  

The above architectures are downloaded and stored for you in the `/data/bottleneck_features/` folder.

This means the following will be in the `/data/bottleneck_features/` folder:

`DogVGG19Data.npz`
`DogResnet50Data.npz`
`DogInceptionV3Data.npz`
`DogXceptionData.npz`



### (IMPLEMENTATION) Obtain Bottleneck Features

In the code block below, extract the bottleneck features corresponding to the train, test, and validation sets by running the following:

    bottleneck_features = np.load('/data/bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']''')


code='''### TODO: Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('/data/bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']
'''
st.code(code)


st.write('''### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        <your model's name>.summary()
   
__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.
''')


st.write('''__Answer:__
This time I am using a pre-trained neural network called ResNet50 that was trained on
          over a million images over 50 layers. 

After initializing the model with the Sequential function, I'm adding a global average pooling
          2D layer to reduce the spatial dimensions from the pre-trained model.
The next and final layer is a Dense layer with 133 units, which is the number of dog breeds
          and a softmax activation function. This last layer assists with final classification.''')

code='''### Define your architecture.
from keras.models import Sequential
resnet50_model = Sequential()
resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
resnet50_model.add(Dense(133, activation='softmax'))

resnet50_model.summary()'''
st.code(code)


st.write('''#### Compile the Model''')

st.write('''### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.  

You are welcome to [augment the training data]
         (https://blog.keras.io/building-powerful-image-classification-models-using-very-little-
         data.html), 
         but this is not a requirement. ''')

code='''### TODO: Train the model.
checkpointer = ModelCheckpoint(filepath = 'saved_models/weights.best.Resnet50.hdf5',
                              verbose=1, save_best_only=True)

resnet50_model.fit(train_Resnet50, train_targets,
                validation_data = (valid_Resnet50, valid_targets),
                epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)'''
st.code(code)

st.image('images/dog_breed/model_training_3.png')

st.write('''#### Load the Model with the Best Validation Loss''')

code='''# Load the model weights with the best validation loss.
resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')'''
st.code(code)


st.write('''#### Test the Model

Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater
          than 60%.''')

code='''### TODO: Calculate classification accuracy on the test dataset.
resnet50_predictions = [np.argmax(resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]

test_accuracy = 100 * np.sum(np.array(resnet50_predictions)==np.argmax(test_targets, axis=1))/len(resnet50_predictions)
print(f'Test accuracy: {test_accuracy:.4}%')'''

st.code(code)

st.write('''Test accuracy: 78.3493%''')



st.write('''#### Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`,
          `Afghan_hound`, etc) that is predicted by your model.  

Similar to the analogous function in Step 5, your function should have three steps:
1. Extract the bottleneck features corresponding to the chosen CNN model.
2. Supply the bottleneck features as input to the model to return the predicted vector.
          Note that the argmax of this prediction vector gives the index of the predicted dog
          breed.
3. Use the `dog_names` array defined in Step 0 of this notebook to return the corresponding
          breed.

The functions to extract the bottleneck features can be found in
          `extract_bottleneck_features.py`, and they have been imported in an earlier code cell.
           To obtain the bottleneck features corresponding to your chosen CNN architecture,
          you need to use the function

    extract_{network}
    
where `{network}`, in the above filename, should be one of `VGG19`, `Resnet50`, `InceptionV3`,
          or `Xception`.''')

code='''# Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
def resnet50_predict_breed(img_path):
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = resnet50_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]

'''
st.code(code)

code='''resnet50_predict_breed(test_files[10])'''
st.code(code)

st.image('images/dog_breed/prediction_1.png') 


st.write('''---
### Step 6: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are __required__ to use your CNN from Step 5 to predict dog breed.  

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_output.png)


#### Write your Algorithm''')

code='''### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.


def mobile_app(img_path):
    img = mpimg.imread(img_path)
    imgplot = plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    if dog_detector(img_path) ==1:
        predict_now = resnet50_predict_breed(img_path)[7:]
        print(f'Another cute dog!  You are most probably a {predict_now}.')

    elif face_detector(img_path) >0:
        predict_again = resnet50_predict_breed(img_path)[7:]
        print(f'Hello human, if you were a dog in a previous life, ,you would most probably be a {predict_again}.')

    else:
        print(f'. . . I do not understand. Who or better yet what are you exactly?!')
    '''
st.code(code)


