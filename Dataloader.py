import glob
from re import VERBOSE
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from Deeplabv3plus import DeeplabV3Plus
from utils import make_folder
import cv2
import os

from natsort import natsorted
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image as pil_image

train_paths = sorted(filter(os.path.isfile,glob.glob('train_img/' + '*') ) )
mask_paths = sorted(filter(os.path.isfile,glob.glob('train_label/' + '*') ) )

valid_paths = sorted(filter(os.path.isfile,glob.glob('val_img/' + '*') ) )
valid_mask_paths = sorted(filter(os.path.isfile,glob.glob('val_label/' + '*') ) )

test_paths = sorted(filter(os.path.isfile,glob.glob('test_img/' + '*') ) )
test_mask_paths = sorted(filter(os.path.isfile,glob.glob('test_label/' + '*') ) )

test_paths = natsorted(test_paths)
test_mask_paths = natsorted(test_mask_paths)


class Datagen(tf.keras.utils.Sequence):
    def __init__(self, train_paths, mask_paths, batch_size=32, n_classes=19, dim = (256,256), n_channels = 3, image_size = 256):
        self.list_IDs = train_paths
        self.image_size = image_size
        self.mask_paths = mask_paths
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.on_epoch_end()
        self._PIL_INTERPOLATION_METHODS = {
            'nearest': pil_image.NEAREST,
            'bilinear': pil_image.BILINEAR,
            'bicubic': pil_image.BICUBIC,
        }

    def __len__(self):
        #number of batches per epoch
        return int(np.floor(len(self.list_IDs)/self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))



    def __getitem__(self, index):
    ##Select a set of
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]     # Generate data
        mask_IDs_temp = [self.mask_paths[k] for k in indexes]
        #print(list_IDs_temp)
        X, y = self.__data_generation(list_IDs_temp, mask_IDs_temp) #to be implemented
        return X, y

    def __data_generation(self, list_IDs_temp, mask_IDs_temp):
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size, *self.dim, 1))
        
        for i, ID in enumerate(list_IDs_temp):
            image = load_img(ID)
            image = image.resize((self.image_size,self.image_size), self._PIL_INTERPOLATION_METHODS['nearest'])
            X[i,] = image
        for i, ID in enumerate(mask_IDs_temp):
            image = tf.io.read_file(ID)
            image = tf.image.decode_png(image, channels=1)
            image.set_shape([None, None, 1])
            image = tf.image.resize(images=image, size=[self.image_size, self.image_size])
            y[i,] = image
        return X,y


training_generator = Datagen(train_paths, mask_paths, batch_size=10)
validation_generator = Datagen(valid_paths, valid_mask_paths, batch_size=10)
test_generator = Datagen(test_paths, test_mask_paths, batch_size=10)

model = DeeplabV3Plus(image_size=256, num_classes=19)
model.summary()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy", tf.keras.metrics.MeanIoU(num_classes=19)],
)

checkpoint_filepath = "model_checkpoint"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

reduceonplateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=7,
        verbose=1.0,
        mode="auto",
        min_delta=0.005,
        cooldown=0,
        min_lr=0.0
    )

# model.fit_generator(generator=training_generator,
#                     validation_data=validation_generator,
#                     use_multiprocessing=False,
#                     epochs = 40,
#                     # initial_epoch=50,
#                     callbacks = [model_checkpoint_callback, reduceonplateau] ,
#                     workers = 6)

use_saved_model = True    
#####################Loading saved model if one exsists
# if not os.path.exists('checkpoint_filepath/saved_model.pb') & use_saved_model:
#     model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ["accuracy"],)
# else:
if use_saved_model:

    # model.load_weights('checkpoint/saved_model.pb') #load the model from file
    model = tf.keras.models.load_model(checkpoint_filepath)
    #loss, acc = model.evaluate_generator(test_generator, steps=3, verbose=0)
    #print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
    make_folder("predictionimages")
    predictions = model.predict_generator(test_generator, steps = 3)
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=3)
    #print(predictions.shape)
    for i, ID in enumerate(predictions):
        #print(ID.shape)
        image = os.path.join("predictionimages", str(i) + '.png')
        print(image)
        cv2.imwrite(image, ID)



#model.save("newmodel")