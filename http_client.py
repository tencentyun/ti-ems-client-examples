import tensorflow as tf
import numpy as np
import requests
import json
from preprocessing import inception_preprocessing

import cv2
import time
import os
#TI-EMS HTTP port is 80 
tf.app.flags.DEFINE_integer('server','localhost','prediction host')
tf.app.flags.DEFINE_string('URL', "http://localhost:9001/v1/models/m:predict", 'http://localhost:80/v1/models/m:predict')
tf.app.flags.DEFINE_string('token', 'sjqfKlGJAayzdRj6TYJYpFzOE5asffF2IumN','TI-EMS access token')
tf.app.flags.DEFINE_string('data_dir', '/data/230', 'path to image in JPEG format')
tf.app.flags.DEFINE_integer('height', 299, 'image height')
tf.app.flags.DEFINE_integer('width', 299, 'image width')
FLAGS = tf.app.flags.FLAGS

def main(_):

  def preprocess_fn(path):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = inception_preprocessing.preprocess_image(image, FLAGS.width, FLAGS.height, is_training=False)
    return image
  
  files = [os.path.join(path, name) for path, _, files in os.walk(FLAGS.data_dir) for name in files]
  headers = {"content-type": "application/json", "X-Auth-Token": FLAGS.token}
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config = config)
  for f in files:
    image = preprocess_fn(f)
    image = sess.run(image)
    image = np.reshape(image,(1, FLAGS.height, FLAGS.width, 3))
    message= image.tolist()
    print(image.dtype)
    print(image.shape)
    body={
#   "signature_name": "serving_default", #xception
    "signature_name": "predict_images",
    "inputs": message }
    start = time.time()
    print("send request")
    result = requests.post("http://"+FLAGS.server+":80/v1/models/m:predict", data=json.dumps(body), headers = headers)
    print(result.text)
    dur1 = time.time() - start
    print("Get Result time: %.6f" % dur1)
if __name__ == '__main__':
  tf.app.run()
