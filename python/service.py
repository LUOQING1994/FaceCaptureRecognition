from flask import Flask,request
import json
import cv2
import base64
import numpy as np

app = Flask(__name__)

flag = 1
@app.route('/upFaceData', methods=['post'])
def upFaceData():
    global flag
    if request.method == 'POST':
        filename = request.files
        if flag == 0:
            return
        if (isinstance(filename, str)):
            filname1 = json.loads(filename)
            tmp = filname1['file']
            img = base64.b64decode(str(tmp))
            image_data = np.fromstring(img, np.uint8)
            image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            cv2.imwrite('01.png', image_data)
        else:
            tmp = filename['file']
            img = base64.b64decode(str(tmp))
            image_data = np.fromstring(img, np.uint8)
            image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            cv2.imwrite('02.png', image_data)
        flag = 0
    return '200'


if __name__ == '__main__':
    app.run()