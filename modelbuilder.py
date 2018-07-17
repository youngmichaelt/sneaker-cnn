import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.models import load_model
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import keras

#set order for data
from keras import backend as K
K.set_image_dim_ordering('th')

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam

PATH = os.getcwd()
# Define data path
data_path = PATH + '//data'
data_dir_list = os.listdir(data_path)

img_rows = 96
img_cols = 96
num_channel = 1
num_epoch = 100

num_classes = 19

#store img data
img_data_list = []

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'//'+dataset)
    print('Loaded images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        #read image
        input_img = cv2.imread(data_path + '//' + dataset + '//' + img)

      
        #resize
        input_img_resize=cv2.resize(input_img,(96,96))
 
        #append to list
        img_data_list.append(input_img_resize)
        

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')

img_data /= 255




if K.image_dim_ordering()=='th':
    img_data = np.rollaxis(img_data, 3,1)



#define number of classes
num_classes = 19

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')


labels[0:468]=0
labels[468:1053]=1
labels[1053:1456]=2
labels[1456:1832]=3
labels[1832:2273]=4
labels[2273:2845]=5
labels[2845:3302]=6
labels[3302:3841]=7
labels[3841:4431]=8
labels[4431:5260]=9
labels[5260:5912]=10
labels[5912:6485]=11
labels[6485:7022]=12
labels[7022:7472]=13
labels[7472:8038]=14
labels[8038:8666]=15
labels[8666:9231]=16
labels[9231:9690]=17
labels[9690:10402]=18
	  
names = ['blank canvas','chalk coral','core black', 'multicolor', 'pharrell oreo', 'pale nude', 'pink glow', 'sun glow','beluga','beluga 2.0', 'blue tint', 'bred', 'butter', 'copper', 'cream',
         'frozen yellow', 'oreo', 'waverunner','zebra']


# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#shuffle dataset
x,y = shuffle(img_data,Y, random_state=2)

#split dataset
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=2)


#building cnn

input_shape = img_data[0].shape


model = Sequential()
	    
 
# if we are using "channels first", update the input shape
# and channels dimension

			
chanDim = -1

    
# CONV => RELU => POOL
model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=(3,96,96)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3), dim_ordering='th'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
##model.add(Dropout(0.5))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
##model.add(Dropout(0.5))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation("softmax"))


aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
opt = Adam(lr=0.001, decay=0.001 / num_epoch)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=["accuracy"])

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

BS = 256
          
data = aug.flow(X_train, y_train, batch_size = BS)

hist = model.fit_generator(
	data,
	validation_data=(X_test, y_test),
	steps_per_epoch=len(X_train) // BS,
	epochs=num_epoch, verbose=2)


model.save('model(256100)')

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])

plt.style.use(['classic'])
plt.show()

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)

plt.style.use(['classic'])
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]


print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])

from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
##y_pred = model.predict_classes(X_test)
##print(y_pred)
target_names = ['class 0(blank canvas)', 'class 1(chalk coral)',
                'class 2(core black)', 'class 3(multicolor)',
                'class 4(pharrell oreo)', 'class 5(pale nude)',
                'class 6(pink glow)', 'class 7(sun glow)',
                'class 8(beluga)', 'class 9(beluga 2.0)',
                'class 10(blue tint)','class 11(bred)',
                'class 12(butter)', 'class 13(copper)',
                'class 14(cream)', 'class 15(frozen yellow)',
                'class 16(oreo)', 'class 17(waverunner)',
                'class 18(zebra)']
					
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

##test_image = cv2.imread('/Users/mikey/pyprojects/reinforcement learning/keras/sneakermodellarger/test/copper.png')
####test_image = cv2.imread('C:/Users/mikey/Desktop/dogpic2.jpg')
##
##test_image=cv2.resize(test_image,(96,96))
##test_image = np.array(test_image)
##test_image = test_image.astype('float32')
##test_image /= 255
####print (test_image.shape)
##test_image=np.rollaxis(test_image,2,0)
##test_image= np.expand_dims(test_image, axis=0)
##print(test_image.shape)   

		
# Predicting the test image
##print((model.predict(test_image)))
##print(model.predict_classes(test_image))
