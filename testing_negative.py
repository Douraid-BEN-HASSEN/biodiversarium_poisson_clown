import tensorflow as tf
import numpy as np

class_names = ['didou', 'eliot', 'kawtar', 'kevin', 'omaima', 'shark']
#class_names = ['didou', 'eliot', 'kevin', 'omaima', 'shark']

img_height = 180
img_width = 180

model = tf.keras.models.load_model('./protocole_1/protocole_1_200ep.hdf5')
#model = tf.keras.models.load_model('./best_model.hdf5')

bonne_reponse = 0
mauvaise_reponse = 0

for i in range(1, 51):
    with tf.device('/gpu:0'):
        img = tf.keras.utils.load_img(
            "./test_img/negative/negative_{}.jpg".format(i), target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)

        score = tf.nn.softmax(predictions[0])

        print(
            "[{}] classe trouvé => {} | score => {:.2f}"
            .format(i, class_names[np.argmax(score)], 100 * np.max(score))
        )
        if np.max(score) <= 0.5 or np.max(score) == 1:
            bonne_reponse += 1
        else:
            mauvaise_reponse += 1

print()
print('bon rep => {} | mauvaise rep => {}'.format(bonne_reponse, mauvaise_reponse))
print("SCORE => {}".format((bonne_reponse/(len(class_names)*10)*100)))