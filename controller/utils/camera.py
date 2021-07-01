import cv2
import threading

# video_process
import argparse
import numpy as np
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from PIL import Image, ImageDraw, ImageFont
import time
import pygame
import multiprocessing

Net = cv2.dnn.readNet('models/face_mask_detection.caffemodel', 'models/face_mask_detection.prototxt')

# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# used to record the time when we processed last frame
start_time = 0

# used to record the time at which we processed current frame
end_time = 0

#Calculate and store FPS
fps = 0
real_fps = 0
num_frames = 10
index = 0

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
id2chiclass = {0: 'Wear Mask', 1: 'Not Wear Mask!'}
colors = ((0, 255, 0), (255, 0 , 0))
flag_p = 1

def play_sound():
    file = r'..\utils\NovemberRain1992.mp3'  # Pay attention to the file path and set the MP3 file you need to play
    #pygame.mixer.init()
    print("Play Music 1")
    #track = pygame.mixer.music.load(file)
    #pygame.mixer.music.play()
    #time.sleep(10)
    #pygame.mixer.music.stop()

def puttext_chinese(img, text, point, color):
    pilimg = Image.fromarray(img)
    draw = ImageDraw.Draw(pilimg)  # Print text on the picture
    fontsize = int(min(img.shape[:2])*0.04)
    font = ImageFont.truetype("simhei.ttf", fontsize, encoding="utf-8")
    y = point[1]-font.getsize(text)[1]
    if y <= font.getsize(text)[1]:
        y = point[1]+font.getsize(text)[1]
    draw.text((point[0], y), text, color, font=font)
    img = np.asarray(pilimg)
    return img

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def inference(image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(160, 160), draw_result=True, chinese=False):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=target_shape)
    net=cv2.dnn.readNet('models/face_mask_detection.caffemodel', 'models/face_mask_detection.prototxt')
    net.setInput(blob)
    y_bboxes_output, y_cls_output = net.forward(getOutputsNames(net))
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    # keep_idxs  = cv2.dnn.NMSBoxes(y_bboxes.tolist(), bbox_max_scores.tolist(), conf_thresh, iou_thresh)[:,0]
    tl = round(0.002 * (height + width) * 0.5) + 1  # line thickness
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)
        if draw_result:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[class_id], thickness=tl)

            if class_id==1:
                global flag_p
                if flag_p==1:
                    p = multiprocessing.Process(target=play_sound)
                    p.start()
                    flag_p=0

            if chinese:
                image = puttext_chinese(image, id2chiclass[class_id], (xmin, ymin), colors[class_id])  ###puttext_chinese
            else:
                text = "%s: %.2f, FPS: %.2f" % (id2class[class_id], conf, real_fps)
                cv2.putText(image, text, (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[class_id])
                
    return image


class RecordingThread(threading.Thread):
    def __init__(self, name, camera):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True

        self.cap = camera
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter('./static/video.avi', fourcc, 20.0, (640, 480))

    def run(self):
        while self.isRunning:
            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = inference(frame, target_shape=(260, 260), conf_thresh=0.5)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if ret:
                self.out.write(frame)

        self.out.release()

    def stop(self):
        self.isRunning = False

    def __del__(self):
        self.out.release()

class VideoCamera(object):
    def __init__(self):
        # Turn on the camera, 0 represents the built-in camera of the notebook
        self.cap = cv2.VideoCapture(0)

        # Initialize the video recording environment
        self.is_record = False
        self.out = None

        # Video recording thread
        self.recordingThread = None

    # Exit the program to release the camera
    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = inference(frame, target_shape=(260, 260), conf_thresh=0.5)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        global fps
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        #print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))  
        global real_fps, num_frames, index, start_time, end_time
        index += 1
        if(index >= num_frames):
            index = 0
            end_time = time.time()
            seconds = end_time - start_time
            start_time = end_time
            real_fps = num_frames/seconds
            #print("Real frames per second after calculated FaceMask : {0}".format(real_fps))

        if ret:
            ret, jpeg = cv2.imencode('.jpg', frame)

            # Video recording
            if self.is_record:
                if self.out == None:
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    self.out = cv2.VideoWriter('video.avi', fourcc, 20.0, (640, 480))

                ret, frame = self.cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = inference(frame, target_shape=(260, 260), conf_thresh=0.5)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if ret:
                    self.out.write(frame)
            else:
                if self.out != None:
                    self.out.release()
                    self.out = None

            return jpeg.tobytes()

        else:
            return None

    def start_record(self):
        self.is_record = True
        self.recordingThread = RecordingThread("Video Recording Thread", self.cap)
        self.recordingThread.start()

    def stop_record(self):
        self.is_record = False

        if self.recordingThread != None:
            self.recordingThread.stop()
