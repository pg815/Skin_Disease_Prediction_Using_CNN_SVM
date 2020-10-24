from keras.models import Sequential
from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

class Model:
    def createAndSaveModel(self):
        classifier = Sequential()

        # add convolution
        classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

        # add max pooling
        classifier.add(MaxPooling2D(pool_size=(2, 2)))

        classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

        classifier.add(Dropout(2.5))

        # add flattening
        classifier.add(Flatten())

        # full connection

        # hidden layer
        classifier.add(Dense(units=128, activation='relu'))

        # output layer
        classifier.add(Dense(units=9, activation='softmax'))

        # compile the layers
        # create the model
        classifier.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

        from keras.preprocessing.image import ImageDataGenerator

        # 0 - 1
        # RGB = 255, 0, 0 => 1, 0, 0
        x_train_generator = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

        x_test_generator = ImageDataGenerator(rescale=1. / 255)

        # generate train and test data
        x_train = x_train_generator.flow_from_directory('/home/girish/PycharmProjects/SkinDiseaseDetection/dataset/train',
                                                        target_size=(64, 64), batch_size=30, class_mode='categorical',
                                                        color_mode="rgb")

        x_test = x_test_generator.flow_from_directory('/home/girish/PycharmProjects/SkinDiseaseDetection/dataset/test',
                                                      target_size=(64, 64), batch_size=30, class_mode='categorical')

        # fit the images to thmagese model
        classifier.fit_generator(x_train, steps_per_epoch=15, epochs=30, validation_data=x_test, validation_steps=10)

        # save the model
        json = classifier.to_json()

        file = open('my_model.json', 'w')

        file.write(json)

        file.close()

        # save the weights
        classifier.save_weights('weights.h5', True)


model = Model()
model.createAndSaveModel()