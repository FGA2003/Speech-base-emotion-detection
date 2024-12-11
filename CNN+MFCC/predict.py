import os

import tensorflow as tf
from tensorflow import keras
new_model = tf.keras.models.load_model('Saved_model/model.keras')

# Show the model architecture
new_model.summary()

#Test on testset
score = new_model.evaluate(x_testcnn, y_test, verbose=0)
print("%s: %.2f%%" % (new_model.metrics_names[1], score[1]*100))