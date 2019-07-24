import base64
import json
import requests

URL = 'http://140.143.219.94:80/v1/models/m:predict'
image = '/data/229_230/n02105641_11015.JPEG'
token = 'W6ptf1K4LrtPtN5Ql2MsWUpO0BGQeJa0nHmu'

data = base64.b64encode(open(image,'rb').read()).decode("utf-8")
headers = {
"content-type": "application/json",
"X-Auth-Token": token
}
body={
"signature_name": "serving_default",
"inputs": [{"b64": data}]
}
r= requests.post(URL, data=json.dumps(body), headers = headers)
print(r.text)
