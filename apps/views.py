from django.shortcuts import render

# Create your views here.

import cv2
import datetime
import base64
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators import gzip

thres = 0.75
objects = ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "person"]

classNames = []
classFile = "static/object.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "static/ssd.pbtxt"
weightsPath = "static/graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

objectCounts = {}
captured_images = []
latest_detected_img = None

@gzip.gzip_page
def index(request):
    return render(request, 'index1.html')

@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@gzip.gzip_page
def latest_detected(request):
    global captured_images

    ret, buffer = cv2.imencode('.jpg', latest_detected_img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    captured_images.insert(0, encoded_image)
    if len(captured_images) > 10:
        captured_images = captured_images[:10]

    return JsonResponse(captured_images, safe=False)

def getObjects(img, thres, nms, objects=[], draw=True, save=False):
    global latest_detected_img

    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=1)
                    if save:
                        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %I-%M-%S %p')
                        if className in objectCounts:
                            objectCounts[className] += 1
                        else:
                            objectCounts[className] = 1
                        count = objectCounts[className]
                        filename = f"object_{timestamp}_{count}.jpg"
                        latest_detected_img = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
    return img, objectInfo

def gen_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    fps = cap.get(cv2.CAP_PROP_FPS)
    ffp = fps/3

    count = 0

    #objects = ["bicycle", "car", "motorcycle"]

    while True:
        success, img = cap.read()
        if count == 0:
            result, objectInfo = getObjects(img, thres, 0.2, objects=objects, save=True)
        count = (count + 1) % ffp
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

