from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import math
from gtts import gTTS
from playsound import playsound
import os

lang = 'en'

model = YOLO("yolo-Weights/yolov8n.pt")
    
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"
                ]

app = Flask(__name__)
camera = cv2.VideoCapture(0)  

def generate_frames():
    while True:
        success, frame = camera.read()
        frame = cv2.flip(frame, 1)
        if not success:
            break
        else:
            
            ret, buffer = cv2.imencode('.jpg', frame)
            results = model(frame, stream = True)
            for r in results:
                boxes = r.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    cv2. rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    
                    confidence = math.ceil((box.conf[0]*100))/100
                    
                    cls = int(box.cls[0])
                    text = classNames[cls]
                    
                    speech = gTTS(text = text, lang = lang, slow = False, tld = "com.au")
                    
                    files = os.listdir("C:/Users/Sai/Desktop/flask final/sound")
                    
                    if(f"{text}.mp3") not in files:
                        speech.save(f"C:/Users/Sai/Desktop/flask final/sound/{text}.mp3")
                    
                    playsound(f'C:/Users/Sai/Desktop/flask final/sound/{text}.mp3')
                    
                    print(classNames[cls])
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    
                    cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)
                    
            cv2.imshow('webcam', frame)
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
