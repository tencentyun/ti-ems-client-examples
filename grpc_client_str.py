import grpc
import requests
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf

server = '62.234.200.215:9000'
image = '/data/229_230/n02105641_11015.JPEG'
token = 'O3IxeDLKqv6WWG0dj7RHj0xUTUakFpSH9Y6r'
with open(image, 'rb') as f:
  data = f.read()

channel = grpc.insecure_channel(server)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
# Send request
request = predict_pb2.PredictRequest()
request.model_spec.name = 'm'
request.model_spec.signature_name = 'serving_default'
request.inputs['image_bytes'].CopyFrom(
#request.inputs['images'].CopyFrom(
  tf.contrib.util.make_tensor_proto(data, shape=[1]))
result = stub.Predict(request, 100.0, metadata=(
  ('x-auth-token', token),
  ))  # 100 seconds
print(result)
