import streamlit as st

# st.write('''#### Work in Progress . . . ''')

# st.image('images/atlantis.png')

st.write('''## Convolutional Neural Networks and an Algorithm for a Dog Identification App ''')

st.write('''In this notebook, I will make the first steps towards developing
          an algorithm that could be used as part of a mobile or web app.
          At the end of this project, the code will accept any user-supplied image as input.
          If a dog is detected in the image, it will provide an estimate of the dog's breed.
          If a human is detected, it will provide an estimate of the dog breed that
          is most resembling.
          The image below displays potential sample output of the finished project''')



left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('images/sample_dog_output.png')


st.write('''## Step 0: Import Datasets

### Import Dog Dataset

In the code cell below, I import a dataset of dog images.
           A few variables are populated through the use of the `load_files` function from the scikit-learn library:
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
train_files, train_targets = load_dataset('/data/dog_images/train')
valid_files, valid_targets = load_dataset('/data/dog_images/valid')
test_files, test_targets = load_dataset('/data/dog_images/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("/data/dog_images/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
'''

st.code(code, line_numbers=True) 

def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets




# # load train, test and validation datasets 
# train_files, train_targets = load_dataset('/data/dog_images/train')
# valid_files, valid_targets = load_dataset('/data/dog_images/valid')
# test_files, test_targets = load_dataset('/data/dog_images/test')

# # load list of dog names
# dog_names = [item[20:-1] for item in sorted(glob("/data/dog_images/train/*/"))]

# # print statistics about the dataset
# st.write('There are %d total dog categories.' % len(dog_names))
# st.write('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
# st.write('There are %d training dog images.' % len(train_files))
# st.write('There are %d validation dog images.' % len(valid_files))
# st.write('There are %d test dog images.'% len(test_files)) 


