import operator
import numpy as np
import keras
from keras.preprocessing import image

class Disease:
    def classify(self,testImageFile):
        from keras.models import model_from_json

        # read the json model
        file = open('my_model.json', 'r')
        data = file.read()
        #print(data)

        file.close()

        # classifier will load the model from the data
        # data -> contents of the my_model.json file
        classifier = model_from_json(data)

        # load waits
        classifier.load_weights('weights.h5')

        # load the test image
        from keras.preprocessing import image

        test_image = image.load_img(testImageFile, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        result = classifier.predict(test_image)
        prediction ={
            'Acne Back': result[0][0],
            'Amelanotic Melanoma': result[0][1],
            'Atypical melanocytic naevus': result[0][2],
            'Atypical mycobacterial infection': result[0][3],
            'Basal cell carcinoma': result[0][4],
            'Dermatofibroma': result[0][5],
            'Epidermoid and trichilemmal cyst': result[0][6],
            'Facial acne': result[0][7],
            'Impetigo': result[0][8]
                         }
        prediction = sorted(prediction.items(), key= operator.itemgetter(1), reverse= True)

        return (prediction)