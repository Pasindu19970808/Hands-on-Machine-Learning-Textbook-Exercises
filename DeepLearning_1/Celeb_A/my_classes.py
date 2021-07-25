import numpy as np
from tensorflow import keras
import tensorflow as tf
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator_custom(keras.utils.Sequence):
    def __init__(self,list_filepaths,labels,batch_size = 10,dim = (224,224),n_channels = 3, n_classes = 2, shuffle = True):
        #initialization
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_filepaths = list_filepaths
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.resize_model = tf.keras.models.Sequential()
        self.resize_model.add(tf.keras.layers.experimental.preprocessing.Resizing(height = 224, width = 224))
        self.on_epoch_end()
    def on_epoch_end(self):
        #shuffling the entire training dataset after each epoch
        self.indexes = np.arange(len(self.list_filepaths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __len__(self):
        #calculates number of steps per epoch
        return int(np.floor(len(self.list_filepaths)/self.batch_size))
    def __getitem__(self,index):
        #returning indexes at the start of the new batch
        indexes = self.indexes[index*self.batch_size:(index + 1)*self.batch_size]
        #Find list of IDs
        list_filepaths_temp = [self.list_filepaths[index] for index in indexes]

        X,y = self.__data_generation(list_filepaths_temp)

        return X,y
    def __data_generation(self,list_filepaths_temp):
        X = np.empty((self.batch_size,*self.dim,self.n_channels))
        y = np.empty((self.batch_size),dtype = int)

        #Generate data
        for i,filepath in enumerate(list_filepaths_temp):
            X[i,] =  self.convert_image_to_pixels(filepath)
            y[i] = self.labels[filepath]

        if self.n_classes > 2:
            return X,keras.utils.to_categorical(y,num_classes=self.n_classes)
        else:
            return X,y
    def convert_image_to_pixels(self, filepath):
        #due to resnet-50 working on images of shape(224,224,3), we need to resize the image
        image = tf.io.read_file(filepath)
        image_pixels = tf.io.decode_image(image, channels = 3, dtype = tf.float32)
        return self.resize_model(image_pixels)
        