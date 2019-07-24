import tensorflow as tf
import numpy as np
import requests
import json

import cv2
import time
import os
#TI-EMS HTTP port is 80 
tf.app.flags.DEFINE_string('URL', "http://106.52.225.142:9001/v1/models/m:predict", 'http://localhost:80/v1/models/m:predict')
tf.app.flags.DEFINE_string('token', 'sjqfKlGJAayzdRj6TYJYpFzOE5asffF2IumN','TI-EMS access token')
tf.app.flags.DEFINE_string('data_dir', '/data/230', 'path to image in JPEG format')
tf.app.flags.DEFINE_integer('height', 299, 'image height')
tf.app.flags.DEFINE_integer('width', 299, 'image width')
FLAGS = tf.app.flags.FLAGS

def main(_):

  files = [os.path.join(path, name) for path, _, files in os.walk(FLAGS.data_dir) for name in files]
  total = 0
  headers = {"content-type": "application/json", "X-Auth-Token": FLAGS.token}
  for f in files:
    image = cv2.imread(f, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (FLAGS.height,FLAGS.width), interpolation=cv2.INTER_CUBIC)
    image = image.transpose((2, 0, 1))
    #if your model input layout is not BGR, please change it to RGB 
    #B, G, R = image
    #image = np.array((R,G,B), dtype=np.uint8)
    c, h, w = image.shape
    message= np.reshape(image,(1, c, h, w)).tolist()
    print(image.shape)
    body={
    "signature_name": "predict_images",
    "inputs": message }
    start = time.time()
    print("send request")
    result = requests.post(FLAGS.URL, data=json.dumps(body), headers = headers)
    print(result.text)
    dur1 = time.time() - start
    print("Get Result time: %.6f" % dur1)
if __name__ == '__main__':
  tf.app.run()
