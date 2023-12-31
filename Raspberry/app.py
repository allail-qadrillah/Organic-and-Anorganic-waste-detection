from tensorflow.lite.python.interpreter import Interpreter
import RPi.GPIO as GPIO
import numpy as np
import time
import cv2

class Servo:
    def __init__(self, servo_pin:int, pwm_period:int=50):
        self.servo_pin = servo_pin
        self.pwm_period= pwm_period
    
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.servo_pin, GPIO.OUT)
        GPIO.setwarnings(False)

        self.pwm = GPIO.PWM(self.servo_pin, self.pwm_period)
        self.pwm.start(7)

    def rotate_180(self):
        "rotate servo to 180 degrees"
        self.pwm.ChangeDutyCycle(2.0)
        time.sleep(0.5)
        self.pwm.ChangeDutyCycle(12.0)
        time.sleep(0.5)

        GPIO.cleanup()
class ObjectDetection():
    def __init__(self, model_path:str, webcam_path:str, label_path:str):
        self.model_path = model_path
        self.webcam_path = webcam_path
        self.label_path = label_path

    def start(self, min_conf=0.5):
        # Load the label map into memory
        with open(self.label_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Load the Tensorflow Lite model into memory
        interpreter = Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()

        # Get model details
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()

        height = input_details['shape'][1]
        width = input_details['shape'][2]
        float_input = (input_details['dtype'] == np.float32)
        input_mean = 127.5
        input_std = 127.5

        # Open video capture
        cap = cv2.VideoCapture(self.webcam_path)

while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to expected shape [1xHxWx3]
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imH, imW, _ = frame.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    detections = []


# Loop over all detections and draw detection box if confidence is above minimum threshold
for i in range(len(scores)):
    if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
        # Get bounding box coordinates and draw box
        ymin = int(max(1,(boxes[i][0] * imH)))
        xmin = int(max(1,(boxes[i][1] * imW)))
        ymax = int(min(imH,(boxes[i][2] * imH)))
        xmax = int(min(imW,(boxes[i][3] * imW)))
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

        # Draw label
        object_name = labels[int(classes[i])]
        label = '%s: %d%%' % (object_name, int(scores[i]*100))
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_ymin = max(ymin, labelSize[1] + 10)
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        detections.append(object_name.upper())
            
            # IF ORGANIK ROTATE SERVO
            organik = ['AMPAS TEBU', 'RANTING KAYU', 'DAUN', 'KULIT TELUR']
            if organik in detections:
                print('ROTATE SERVO')
                       
            # Show the output frame
            cv2.imshow('KLasifikasi Sampah Organik dan Anorganik', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
            
            print(detections)


        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        return detections

if __name__ == '__main__':
  # Set up variables for running user's model
  PATH_TO_VIDEO = './video-test.mp4' #0 #'./video-test.mp4'      # Path to Video Capture 
  PATH_TO_MODEL = './model/detect.tflite' # Path to model tflite
  PATH_TO_LABELS = './model/labelmap.txt' # Path to labelmap.txt file

  # Run inferencing function!
  app = ObjectDetection(
      model_path=PATH_TO_MODEL,
      webcam_path=PATH_TO_VIDEO,
      label_path=PATH_TO_LABELS
  )

  # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
  min_conf_threshold = 0.5

  app.start(min_conf=min_conf_threshold)