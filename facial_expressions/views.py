from django.shortcuts import render,redirect
from django.http import StreamingHttpResponse
from django.views.decorators import gzip

# Create your views here.
import cv2
import threading
import os
import time
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import base64
import io
from io import BytesIO
from PIL import Image

#import tensorflow as tf
#from keras.models import model_from_json
class FacialExpressionModel():
    emotions_list=["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"] 
                     
    def __init__(self,model_json_file,model_weights_file):
        with open(model_json_file,'r') as json_file:
            self.loaded_model = model_from_json(json_file.read())
        self.loaded_model.load_weights(model_weights_file)
    
    def predict_emotion(self,img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.emotions_list[np.argmax(self.preds)]

               

base_path=os.path.abspath(os.path.dirname(__file__))





class VideoCamera():
    def __init__(self):
        self.facec = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.model = FacialExpressionModel(os.path.join(base_path,'model/model.json'),os.path.join(base_path,'model/model_weights.h5'))
        self.video = cv2.VideoCapture(0)
        (self.grabbed,self.frame) = self.video.read()
        threading.Thread(target=self.update,args=()).start()
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        #(self.grabbed,self.frame) = self.video.read()
        
        gray_fr = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self.facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = self.model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(self.frame, pred, (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,0,0),2)
         
        _,jpeg = cv2.imencode('.jpg',self.frame)
        return jpeg.tobytes()
    
    def update(self):
        while True:
            (self.grabbed,self.frame) = self.video.read()
            
            
# \r is the carriage return -escape character. It shifts the cursor to the beginning of string of line
# and replaces all characters in beginning by the ones after the carriage return.
def gen(camera):
     while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n\r\n')
        
        
#@gzip.gzip_page
def live(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam),content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print('aborted')                                   
        
def facial_expression_detection(request):
    return render(request,'facial_expressions/camera.html')  

def base(request):
    return render(request,'facial_expressions/base.html')  
    
def upload(request):
    return render(request,'facial_expressions/upload.html')    

def to_data_uri(pil_image):
    data = BytesIO()
    pil_image.save(data,'JPEG')
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8')

def to_image(numpy_img):
    return Image.fromarray(numpy_img,'RGB')

def recognize(request):
    img = request.FILES['image']
    img_file = io.BytesIO(img.read())
    img_pil = Image.open(img_file)
    frame = img_to_array(img_pil,data_format='channels_last',dtype='uint8')
        

    facec = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    model = FacialExpressionModel(os.path.join(base_path,'model/model.json'),os.path.join(base_path,'model/model_weights.h5'))

    gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr,1.5,5)
    for (x, y, w, h) in faces:
        fc = gray_fr[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

        cv2.putText(frame, pred, (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    
    
    
    pil_image = to_image(frame)
    image_uri = to_data_uri(pil_image)

    context = {'image_uri':image_uri}
    return render(request,'facial_expressions/index.html',context)




