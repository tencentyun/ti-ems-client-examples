import grpc
import tensorflow as tf
import numpy as np

import cv2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import time
import os

tf.app.flags.DEFINE_string('server', 'localhost',
                           'TI-EMS serving IP')
tf.app.flags.DEFINE_string('token', 'sjqfKlGJAayzdRj6TYJYpFzOE5asffF2IumN','TI-EMS access token')
tf.app.flags.DEFINE_string('data_dir', '/data/230', 'path to image in JPEG format')
tf.app.flags.DEFINE_integer('height', 299, 'image height')
tf.app.flags.DEFINE_integer('width', 299, 'image width')
FLAGS = tf.app.flags.FLAGS

def main(_):

  files = [os.path.join(path, name) for path, _, files in os.walk(FLAGS.data_dir) for name in files]
  sess = tf.Session()
  for f in files:
    image = cv2.imread(f, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (FLAGS.height,FLAGS.width), interpolation=cv2.INTER_CUBIC)
    image = image.transpose((2, 0, 1))
    #if your model input layout is not BGR, please change it to RGB 
    #B, G, R = image
    #image = np.array((R,G,B), dtype=np.uint8)
    channel = grpc.insecure_channel(FLAGS.server + ":9000")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    c, h, w = image.shape
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'm'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=[1, c, h, w])) #worked
    start = time.time()
    result = stub.Predict(request, 10.0, metadata=(
    ('x-auth-token', FLAGS.token),
    ))  # 10 seconds
#Example code to check the inference accuracy, the method depends on the model and the test dataset
    val = result.outputs['InceptionV4/Logits/Predictions'].float_val
    target = max(val)
    for i in range(0, 1000):
      if (val[i] == target):
        index = i
    print(index)
    dur1 = time.time() - start
    print("Get Result time: %.6f" % dur1)
#acc = total / 200.0
#print("Accuracy: %.4f" % acc)
if __name__ == '__main__':
  tf.app.run()
