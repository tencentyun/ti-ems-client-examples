import grpc
import tensorflow as tf
import numpy as np
import inception_preprocessing

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import time
import os

tf.app.flags.DEFINE_string('server', 'localhost',
                           'PredictionService host')
tf.app.flags.DEFINE_string('token', 'WEI5GUL2Tn575Rf4hQyWEAjfgeKBZZSDkeBx','TI-EMS access token')
tf.app.flags.DEFINE_string('data_dir', '/data/229', 'path to image in JPEG format')
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
  dataset = tf.data.Dataset.from_tensor_slices(files)
  dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=preprocess_fn, batch_size=1, num_parallel_calls=1))
  dataset = dataset.repeat(count=1)
  iterator = dataset.make_one_shot_iterator()
  sess = tf.Session()
  for i in range(1,50):
      image = iterator.get_next()
      image = sess.run(image)
      channel = grpc.insecure_channel(FLAGS.server+":9000")
      stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
      
      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'm'
#request.model_spec.signature_name = 'serving_default'
      request.model_spec.signature_name = 'predict_images'
      request.inputs['image'].CopyFrom(
          tf.contrib.util.make_tensor_proto(image.astype(dtype=np.float32), shape=[1, FLAGS.height, FLAGS.width, 3])) #worked
      start = time.time()
      result = stub.Predict(request, 100.0, metadata=(
('x-auth-token', FLAGS.token),
))  # wait 100 seconds
      print(result.outputs['out'].int64_val)
      dur1 = time.time() - start
      print("Get Result time: %.6f" % dur1)

if __name__ == '__main__':
  tf.app.run()
