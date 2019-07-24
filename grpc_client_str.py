import grpc
import requests
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf

server = '154.8.187.43:9000'
image = '/data/229_230/n02105641_11015.JPEG'
token = 'YT8VpqUm7JZ58vlyt7vghOp69gga0ZN3vhVZ'
with open(image, 'rb') as f:
  data = f.read()

channel = grpc.insecure_channel(server)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
# Send request
request = predict_pb2.PredictRequest()
request.model_spec.name = 'm'
request.model_spec.signature_name = 'serving_default'
request.inputs['images'].CopyFrom(
  tf.contrib.util.make_tensor_proto(data, shape=[1]))
result = stub.Predict(request, 10.0, metadata=(
  ('x-auth-token', token),
  ))  # 10 seconds
print(result)
