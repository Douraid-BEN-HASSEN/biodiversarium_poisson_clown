import tensorflow as tf
import numpy as np

class_names = ['didou', 'eliot', 'kevin', 'omaima', 'shark']

img_height = 180
img_width = 180

model = tf.keras.models.load_model('./protocole_2/protocole_2_200ep.hdf5')
dossier = ['def', 'nor']

bonne_reponse = 0
mauvaise_reponse = 0

with tf.device('/gpu:0'):
    for i in range(1, 2):
        for n in range(1, 26):                
            img = tf.keras.utils.load_img(
                "./test_img/{}/{}{}.JPG".format(i, dossier[i-1], n), target_size=(img_height, img_width)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)

            score = tf.nn.softmax(predictions[0])

            print(
                    "classe trouvé => {} | score => {:.2f}"
                    .format(class_names[np.argmax(score)], 100 * np.max(score))
                )
            if class_names[np.argmax(score)] == 'didou':
                bonne_reponse += 1
            else:
                mauvaise_reponse += 1
        print()

print('+------------------------+')
print('bonne_reponse => {} | mauvaise_reponse => {} | SCORE => {}'.format(bonne_reponse, mauvaise_reponse, (bonne_reponse/25)*100))
print('--------------------------')