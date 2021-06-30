# -*- coding: utf-8 -*-
from flask import session, render_template, redirect, url_for, Response, make_response
from controller.modules.home import home_blu
from controller.utils.camera import VideoCamera
#from smbus2 import SMBus
#from mlx90614 import MLX90614
video_camera = None
global_frame = None

# Home page
@home_blu.route('/')
def index():
    # Template rendering
    username = session.get("username")
    # Get Sensor temperatures
    #bus = SMBus(1)
    #sensor = MLX90614(bus, address=0x5A)
    ambient = round(28.88,1) #round(sensor.get_ambient(),1) 
    temp = round(36.55,1) #round(sensor.get_object_1(),1)
    tempInfo = {
        'ambient' : ambient,
        'temp'    : temp
    }
    #bus.close()

    if not username:
        return redirect(url_for("user.login"))
    return render_template("index.html",**tempInfo)


# Get video stream from camera
def video_stream():
    global video_camera
    global global_frame

    if video_camera is None:
        video_camera = VideoCamera()

    while True:
        frame = video_camera.get_frame()
        if frame is not None:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')


# Video streaming
@home_blu.route('/video_viewer')
def video_viewer():
    # Template rendering
    username = session.get("username")
    # voice_alert()
    if not username:
        return redirect(url_for("user.login"))
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


