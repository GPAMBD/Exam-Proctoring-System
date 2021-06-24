import tensorflow as tf
import numpy as np
import cv2
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import threading
import time
from PIL import Image,ImageTk

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization
)
from tensorflow.keras.regularizers import l2
import wget

import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks

# Some tkinter objects
color_ad = None
kill = False
showWindowVari = 0
showAlerts = 0

def openForm():
    global color_ad
    TitleFont = ("Benne Regular", 35)
    BtnFont = ("Gill Sans MT", 15)
    font_l = ("Berlin Sans FB", 14)
    txtFont = ("Courier New", 15)

    # Function to check the pixel number (Its just to help developer)
    base = Tk()
    base.title("Exam Proctoring Sytem")
    base.state("zoomed")

    # Command Functions

    def resetFields():
        loginIdEntry.delete(0, END)
        passEntry.delete(0, END)
        loginIdEntry.focus()

    def loginFieds():
        login = loginIdEntry.get()
        password = passEntry.get()
        if login == "pranav@1011" and password == "8268":
            messagebox.showinfo("Login Successful", "Login Successful !")
            hideProctorLogin()
            showproctorConsole()
        elif login == "" and password != "":
            messagebox.showerror("Login Failed", "Please Insert You Login Id !")
        elif login != "" and password == "":
            messagebox.showerror("Login Failed", "Please Insert You Password !")
        elif login == "" and password == "":
            messagebox.showerror("Login Failed", "Please Insert You Credentials !")
        elif login != "pranav@1011" and password == "8268":
            messagebox.showerror("Login Failed", "Incorrect Login Id ! Please Try Again !")
        elif login == "pranav@1011" and password != "8268":
            messagebox.showerror("Login Failed", "Incorrect Password ! Please Try Again !")
        elif login != "pranav@1011" and password != "8268":
            messagebox.showerror("Login Failed", "Invalid Credentials ! Please Try Again !")

    def hideMainLobby():
        mainLabel.place_forget()
        textMainLabel.place_forget()
        Lcanvas.place_forget()
        # studLoginBtn.place_forget()
        proctorLoginBtn.place_forget()

    def hideProctorLogin():
        textProctorLabel.place_forget()
        Pcanvas.place_forget()
        loginMiniLabel.place_forget()
        loginIdLabel.place_forget()
        loginIdEntry.place_forget()
        passLabel.place_forget()
        passEntry.place_forget()
        loginBtn.place_forget()
        resetBtn.place_forget()

    # Proctoring Controls
    def startProctoring():
        startBtn.configure(bg="#7FFF00")
        stopBtn.configure(bg="#DC143C")
        # Starting all threads for proctoring
        startThreads()
        newThread = threading.Thread(target=changColorAd)
        newThread.start()
        print("At end of startProctoring Function")

    def stopProctoring():
        global color_ad, kill
        startBtn.configure(bg="#DC143C")
        stopBtn.configure(bg="#7FFF00")
        color_ad = 0
        kill = True

    def refreshStatus():
        global color_ad
        color_ad = 9

    def showWindows():
        global showWindowVari
        print("Showing Windows")
        showWindowVari = 1

    def showAlertsFunc():
        global showAlerts
        if showAlerts == 0:
            messagebox.showinfo("Enable Alerts","Enable Show Alerts")
            showAlerts = 1
        else:
            messagebox.showinfo("Disable Alerts", "Disable Show Alerts")
            print("Hiding Alerts")
            showAlerts = 0


    # END of Command Functions -------------------------------------

    # Proctor Login Section
    def showproctorLogin():
        hideMainLobby()
        mainProctorLabel.place(x=0, y=0)
        textProctorLabel.place(x=569, y=30)
        Pcanvas.place(x=180, y=230)
        loginMiniLabel.place(x=910, y=315)
        loginIdLabel.place(x=809, y=370)
        loginIdEntry.place(x=916, y=377)
        passLabel.place(x=809, y=415)
        passEntry.place(x=916, y=423)
        loginBtn.place(x=819, y=481)
        resetBtn.place(x=980, y=481)

    # END Of Proctor Login Section ---------------------

    # Proctor Console Section
    def showproctorConsole():
        hideProctorLogin()
        hideMainLobby()
        # mainLabel.place(x=0, y=0)  # For Testing Remove it after work done
        proctorConsolMainTextLabel.place(x=530, y=30)

        operationStatusLabel.place(x=171, y=155)
        startBtn.place(x=171, y=240)
        stopBtn.place(x=436, y=240)
        personDetectBtn.place(x=702, y=240)
        noPersonDetectBtn.place(x=969, y=240)
        eyeDetectBtn.place(x=171, y=350)
        poseDetectBtn.place(x=436, y=350)
        phoneDetectBtn.place(x=702, y=350)
        speakDetectBtn.place(x=969, y=350)

        operationControlLabel.place(x=171, y=480)
        startProctoringBtn.place(x=171, y=550)
        refreshStatus.place(x=436, y=550)
        showWindowsBtn.place(x=702, y=550)
        stopProctoringBtn.place(x=969, y=550)
        showAlertBtn.place(x=171, y=629)

    # TEsting feature
    def motion(event):
        x, y = event.x, event.y
        mainLabel.config(text="x = " + str(x) + ", y = " + str(y))

    base.bind('<Motion>', motion)

    # Lobby Section
    mainLabel = Label(base, height=8, width=300, bg="gray")
    mainLabel.place(x=0, y=0)

    textMainLabel = Label(base, fg="white", bg="gray", text="Exam Proctoring System", font=TitleFont)
    textMainLabel.place(x=450, y=30)

    Limg = ImageTk.PhotoImage(Image.open("proctorImgS.jpeg"))
    Lcanvas = Canvas(base, width=460, height=337, bg="gray")
    Lcanvas.create_image(0, 0, anchor=NW, image=Limg)
    Lcanvas.place(x=180, y=230)

    ttk.Style().configure("TButton", padding=12, relief="flat", background="#ccc", font=BtnFont)
    # studLoginBtn = ttk.Button(base, text="Student Login", width=30)
    # studLoginBtn.place(x=816, y=320)

    proctorLoginBtn = ttk.Button(base, text="Proctor Login", width=30, command=showproctorLogin)
    proctorLoginBtn.place(x=816,y=367)

    # Proctor Login Widgets
    mainProctorLabel = Label(base, height=8, width=300, bg="gray")
    textProctorLabel = Label(base, fg="white", bg="gray", text="Proctor Login", font=TitleFont)

    img = ImageTk.PhotoImage(Image.open("proctorImgS.jpeg"))
    Pcanvas = Canvas(base, width=460, height=337, bg="gray")
    Pcanvas.create_image(0, 0, anchor=NW, image=img)
    loginMiniLabel = Label(base, text="Login", font=(txtFont[0], 20))
    loginIdLabel = Label(base, text="Login Id : ", font=(TitleFont[0], 16))
    loginIdEntry = Entry(base, width=20, border=1, font=(txtFont[0], 14))
    passLabel = Label(base, text="Password : ", font=(TitleFont[0], 16))
    passEntry = Entry(base, width=20, border=1, font=(txtFont[0], 14), show="*")
    loginBtn = Button(base, text="Login", width=20, command=loginFieds)
    resetBtn = Button(base, text="Reset", width=20, command=resetFields)

    # Proctor Console Widgets
    proctorConsolMainTextLabel = Label(base, fg="white", bg="gray", text="Proctor Console", font=TitleFont)

    operationStatusLabel = ttk.Label(base, text="Operation Status", font=(TitleFont[0], 30))
    startBtn = Button(base, text="Start Proctoring Status", height=2, width=20, font=BtnFont)
    stopBtn = Button(base, text="Stop Proctoring Status", height=2, width=20, font=BtnFont)
    eyeDetectBtn = Button(base, text="Eye Detection Status", height=2, width=20, font=BtnFont)
    poseDetectBtn = Button(base, text="Pose Detection Status", height=2, width=20, font=BtnFont)
    phoneDetectBtn = Button(base, text="Phone Detection Status", height=2, width=20, font=BtnFont)
    personDetectBtn = Button(base, text="Person Detection Status", height=2, width=20, font=BtnFont)
    noPersonDetectBtn = Button(base, text="No Person Detection", height=2, width=20, font=BtnFont)
    speakDetectBtn = Button(base, text="Speak Detection Status", height=2, width=20, font=BtnFont)

    # operation btns
    operationControlLabel = ttk.Label(base, text="Operation Controls", font=(TitleFont[0], 30))
    startProctoringBtn = ttk.Button(base, text="Start Proctoring", width=20, command=startProctoring)
    refreshStatus = ttk.Button(base, text="Refresh Status", width=20, command=refreshStatus)
    showWindowsBtn = ttk.Button(base, text="Monitor Actions", width=20, command=showWindows)
    stopProctoringBtn = ttk.Button(base, text="Stop Proctoring", width=20, command=stopProctoring)
    showAlertBtn = ttk.Button(base, text="Show Alerts", width=20, command=showAlertsFunc)

    def changColorAd():
        print("\nThread Start...\n")
        while(True):
            if color_ad == 1:
                poseDetectBtn.configure(bg="red")
            elif color_ad == 2:
                phoneDetectBtn.configure(bg="red")
            elif color_ad == 3:
                noPersonDetectBtn.configure(bg="red")
            elif color_ad == 4:
                personDetectBtn.configure(bg="red")
            elif color_ad == 5:
                speakDetectBtn.configure(bg="red")
            elif color_ad == 6:
                eyeDetectBtn.configure(bg="red")
            elif color_ad == 7:
                eyeDetectBtn.configure(bg="red")
            elif color_ad == 8:
                eyeDetectBtn.configure(bg="red")
            elif color_ad == 0:
                eyeDetectBtn.configure(bg=None)
                poseDetectBtn.configure(bg=None)
                phoneDetectBtn.configure(bg=None)
                speakDetectBtn.configure(bg=None)
                noPersonDetectBtn.configure(bg=None)
                personDetectBtn.configure(bg=None)

            else:
                eyeDetectBtn.configure(bg="#7FFF00")
                poseDetectBtn.configure(bg="#7FFF00")
                phoneDetectBtn.configure(bg="#7FFF00")
                speakDetectBtn.configure(bg="#7FFF00")
                noPersonDetectBtn.configure(bg="#7FFF00")
                personDetectBtn.configure(bg="#7FFF00")



    # Executing GUI
    base.mainloop()

# ----- end of tkinter objecyt


def load_darknet_weights(model, weights_file):
    # Open the weights file
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    # Define names of the Yolo layers (just for a reference)
    layers = ['yolo_darknet',
              'yolo_conv_0',
              'yolo_output_0',
              'yolo_conv_1',
              'yolo_output_1',
              'yolo_conv_2',
              'yolo_output_2']

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):

            if not layer.name.startswith('conv2d'):
                continue

            # Handles the special, custom Batch normalization layer
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416

yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


def DarknetConv(x, filters, kernel_size, strides=1, batch_norm=True):

    # Image padding
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'

    # Defining the Conv layer
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)

    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def YoloConv(filters, name=None):

    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):

    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return yolo_output


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    # Extract box coortinates from prediction vectors
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    # Normalize coortinates
    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
             tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=0.5,
        score_threshold=0.6
    )

    return boxes, scores, classes, valid_detections


def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80):
    x = inputs = Input([size, size, channels], name='input')

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


def weights_download(out='models/yolov3.weights'):
    _ = wget.download('https://pjreddie.com/media/files/yolov3.weights', out='models/yolov3.weights')

# ---------- HeadPOse ----------

def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)


def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]

    return (x, y)

# Eye Detector
def eye_on_mask(mask, side, shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1] + points[2][1]) // 2
    r = points[3][0]
    b = (points[4][1] + points[5][1]) // 2
    return mask, [l, t, r, b]


def find_eyeball_position(end_points, cx, cy):
    x_ratio = (end_points[0] - cx) / (cx - end_points[2])
    y_ratio = (cy - end_points[1]) / (end_points[3] - cy)
    if x_ratio > 3:
        return 1
    elif x_ratio < 0.33:
        return 2
    elif y_ratio < 0.33:
        return 3
    else:
        return 0


def contouring(thresh, mid, img, end_points, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        pass


def process_thresh(thresh):
    """
    Preprocessing the thresholded image

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image to preprocess

    Returns
    -------
    thresh : Array of uint8
        Processed thresholded image

    """
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 3)
    thresh = cv2.bitwise_not(thresh)
    return thresh


def print_eye_pos(img, left, right):
    global color_ad
    if left == right and left != 0:
        text = ''
        if left == 1:
            print('Looking left')
            text = 'Looking left'
            color_ad = 6
            if showAlerts == 1:
                loc = threading.Lock()
                msgShow(6, loc)
        elif left == 2:
            print('Looking right')
            text = 'Looking right'
            color_ad = 7
            if showAlerts == 1:
                loc = threading.Lock()
                msgShow(7, loc)
        elif left == 3:
            print('Looking up')
            text = 'Looking up'
            color_ad = 8
            if showAlerts == 1:
                loc = threading.Lock()
                # msgShow(8, loc)
        font_e = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, text, (30, 30), font_e,
        #             1, (0, 255, 255), 2, cv2.LINE_AA)



    # -------------Runner Area ---------------

# --------- eyeTracker ---------


# ----------- HeadPose --------------

# 2 lines for headPose
face_model = get_face_detector()
landmark_model = get_landmark_model()

# 2 lines for mouth
mouth_face_model = get_face_detector()
mouth_landmark_model = get_landmark_model()

# 2 lines for eyeDetector
eye_face_model = get_face_detector()
eye_landmark_model = get_landmark_model()
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]


cap = cv2.VideoCapture(0)
ret, imgView = cap.read()


    # --- msgShower
def msgShow(x,l):
    l.acquire()
    from tkinter import messagebox
    root = Tk()
    root.withdraw()
    # root.withdraw()
    if x == 1:
        messagebox.showwarning("Head Position Detected", "Please keep your face on screen ! Don't Look at Right side.")
    elif x == 2:
        messagebox.showwarning("Mobile Phone Detected", "We are warning you to avoid use of mobile. Otherwise your exam will be closed")
    elif x == 3:
        messagebox.showwarning("No Person Detected !", "Please come on your place...")
    elif x == 4:
        messagebox.showwarning("Don't Cheat", "More than one person detected !")
    elif x == 5:
        messagebox.showwarning("Shut Up !", "Don't Talk With Anyone ! Keep Quite !")
    elif x == 6:
        messagebox.showwarning("Look Straight !", "Who is on you left side ? Don't Look at them !")
    elif x == 7:
        messagebox.showwarning("Look Straight !", "Who is on you right side ? Don't Look at them !")
    elif x == 8:
        messagebox.showwarning("Look Straight !", "Who is on you up side ? Don't Look at them !")
    elif x == 9:
        messagebox.showwarning("Head Position Detected", "Please keep your face on screen ! Don't Look at Left side.")
    elif x == 10:
        messagebox.showinfo("Analysing Your Face...","Please Press key 'c' to continue after clicking on OK.")
    root.destroy()
    l.release()

# EyeTracker
thresh = imgView.copy()
# cv2.namedWindow('image')
e_kernel = np.ones((9, 9), np.uint8)

def nothing():
    pass
# cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

# -- headPose
size = imgView.shape
font = cv2.FONT_HERSHEY_SIMPLEX
# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

# ---- personPhone ------
yolo = YoloV3()
load_darknet_weights(yolo, 'models/yolov3.weights')

# ------ open mouth ------
outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0]*5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0]*3
font = cv2.FONT_HERSHEY_SIMPLEX
font_p = cv2.FONT_HERSHEY_SCRIPT_COMPLEX


# TAking mouth size
while(True):
    ret, img = cap.read()
    rects = find_faces(img, mouth_face_model)
    for rect in rects:
        shape = detect_marks(img, mouth_landmark_model, rect)
        draw_marks(img, shape)
        cv2.putText(img, "Analysing Your Face. Please press button 'c'.", (10, 400), font_p,
                    1, (0,0,0), 2)
        cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        for i in range(100):
            for i, (p1, p2) in enumerate(outer_points):
                d_outer[i] += shape[p2][1] - shape[p1][1]
            for i, (p1, p2) in enumerate(inner_points):
                d_inner[i] += shape[p2][1] - shape[p1][1]
        break
cv2.destroyAllWindows()
d_outer[:] = [x / 100 for x in d_outer]
d_inner[:] = [x / 100 for x in d_inner]

# -------- Threading Function ---------
# Testing Start

# Testing End -----------

def headPoseStarting():
    global color_ad, showWindowVari, showAlerts
    while True:
        try:
            ret, HimgView = cap.read()
            if ret == True:
                faces = find_faces(HimgView, face_model)
                for face in faces:
                    marks = detect_marks(HimgView, landmark_model, face)
                    # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
                    image_points = np.array([
                                            marks[30],     # Nose tip
                                            marks[8],     # Chin
                                            marks[36],     # Left eye left corner
                                            marks[45],     # Right eye right corne
                                            marks[48],     # Left Mouth corner
                                            marks[54]      # Right mouth corner
                                        ], dtype="double")
                    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
                    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)


                    # Project a 3D point (0, 0, 1000.0) onto the image plane.
                    # We use this to draw a line sticking out of the nose

                    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

                    for p in image_points:
                        cv2.circle(HimgView, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


                    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
                    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                    x1, x2 = head_pose_points(HimgView, rotation_vector, translation_vector, camera_matrix)

                    cv2.line(HimgView, p1, p2, (0, 255, 255), 2)
                    cv2.line(HimgView, tuple(x1), tuple(x2), (255, 255, 0), 2)
                    # for (x, y) in marks:
                    #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
                    # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
                    try:
                        m = (p2[1] - p1[1])/(p2[0] - p1[0])
                        ang1 = int(math.degrees(math.atan(m)))
                    except:
                        ang1 = 90

                    try:
                        m = (x2[1] - x1[1])/(x2[0] - x1[0])
                        ang2 = int(math.degrees(math.atan(-1/m)))
                    except:
                        ang2 = 90

                        # print('div by zero error')
                    if ang1 >= 48:
                        print('Head down')
                        # cv2.putText(HimgView, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
                    elif ang1 <= -48:
                        print('Head up')
                        # cv2.putText(HimgView, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)

                    if ang2 >= 48:
                        print('Head right')
                        # cv2.putText(HimgView, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
                        color_ad = 1
                        if showAlerts == 1:
                            loc = threading.Lock()
                            msgShow(1, loc)
                    elif ang2 <= -48:
                        print('Head left')
                        color_ad = 1
                        # cv2.putText(HimgView, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)
                        if showAlerts == 1:
                            loc = threading.Lock()
                            msgShow(9, loc)

                    cv2.putText(HimgView, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
                    cv2.putText(HimgView, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
                if showWindowVari == 1:
                    cv2.imshow('Head Position', HimgView)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Broke head_pose due to 'q' press")
                    break
            else:
                print("Breaking due to ret fail at headPoseDetect..\n")
                break
        except Exception:
            print("Exception occured in Thread-1")
            print("Continuing a thread-1")
            # t1.start()
            continue
        if kill == True:
            print("Thread-1 killed due to you.")
            break

def personDetectStarting():
    global color_ad, showWindowVari
    while (True):
        try:
            ret, PimgView = cap.read()
            if ret == False:
                print("Failed due to ret break at personDetect...\n")
                break
            p_img = cv2.cvtColor(PimgView, cv2.COLOR_BGR2RGB)
            p_img = cv2.resize(p_img, (320, 320))
            p_img = p_img.astype(np.float32)
            p_img = np.expand_dims(p_img, 0)
            p_img = p_img / 255
            class_names = [c.strip() for c in open("models/classes.TXT").readlines()]
            boxes, scores, classes, nums = yolo(p_img)
            count = 0
            for i in range(nums[0]):
                if int(classes[0][i] == 0):
                    count += 1
                if int(classes[0][i] == 67):
                    print('Mobile Phone detected')
                    color_ad = 2
                    if showAlerts == 1:
                        loc = threading.Lock()
                        msgShow(2, loc)
            if count == 0:
                print('No person detected')
                color_ad = 3
                if showAlerts == 1:
                    loc = threading.Lock()
                    msgShow(3, loc)
            elif count > 1:
                print('More than one person detected')
                color_ad = 4
                if showAlerts == 1:
                    loc = threading.Lock()
                    msgShow(4, loc)

            PimgView = draw_outputs(PimgView, (boxes, scores, classes, nums), class_names)

        except Exception:
            print("Exception occured in Thread-2")
            print("Again running a thread-2")
            # t2.start()
            continue
        if showWindowVari == 1:
            cv2.imshow('Person Detection', PimgView)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if kill == True:
            print("Thread-2 killed due to you.")
            break

def mouthDetectStarting():
    global color_ad, showWindowVari
    while (True):
        try:
            ret, MimgView = cap.read()
            rects = find_faces(MimgView, mouth_face_model)
            for rect in rects:
                shape = detect_marks(MimgView, mouth_landmark_model, rect)
                cnt_outer = 0
                cnt_inner = 0
                draw_marks(MimgView, shape[48:])
                for i, (p1, p2) in enumerate(outer_points):
                    if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                        cnt_outer += 1
                for i, (p1, p2) in enumerate(inner_points):
                    if d_inner[i] + 2 < shape[p2][1] - shape[p1][1]:
                        cnt_inner += 1
                if cnt_outer > 3 and cnt_inner > 2:
                    print('Mouth open')
                    color_ad = 5
                    # cv2.putText(MimgView, 'Mouth open', (30, 30), font,
                                # 1, (0, 255, 255), 2)
                    if showAlerts == 1:
                        loc = threading.Lock()
                        msgShow(5, loc)
        except Exception:
            print("Exception occured in Thread-3")
            print("Continuing a thread-3")
            # t3.start()
            continue
        # show the output image with the face detections + facial landmarks
        if showWindowVari == 1:
            MimgView = cv2.resize(MimgView,(440,400), interpolation=cv2.INTER_AREA)
            cv2.imshow("Output", MimgView)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if kill == True:
            print("Thread-3 killed due to you.")
            break

def eyeDetector():
        global color_ad, kill, showWindowVari
        print("I am here in eyeDetector !")
        while (True):
            try:
                ret, EimgView = cap.read()
                rects = find_faces(EimgView, eye_face_model)
                if ret == False:
                    print("Failed due to ret break at eyeDetector...\n")
                    break
                for rect in rects:
                    shape = detect_marks(EimgView, eye_landmark_model, rect)
                    mask = np.zeros(EimgView.shape[:2], dtype=np.uint8)
                    mask, end_points_left = eye_on_mask(mask, left, shape)
                    mask, end_points_right = eye_on_mask(mask, right, shape)
                    mask = cv2.dilate(mask, e_kernel, 5)

                    eyes = cv2.bitwise_and(EimgView, EimgView, mask=mask)
                    mask = (eyes == [0, 0, 0]).all(axis=2)
                    eyes[mask] = [255, 255, 255]
                    mid = int((shape[42][0] + shape[39][0]) // 2)
                    eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                    # threshold = cv2.getTrackbarPos('threshold', 'image')
                    _, thresh = cv2.threshold(eyes_gray, 75, 255, cv2.THRESH_BINARY)
                    thresh = process_thresh(thresh)

                    eyeball_pos_left = contouring(thresh[:, 0:mid], mid, EimgView, end_points_left)
                    eyeball_pos_right = contouring(thresh[:, mid:], mid, EimgView, end_points_right, True)
                    print_eye_pos(EimgView, eyeball_pos_left, eyeball_pos_right)
                    # for (x, y) in shape[36:48]:
                    #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
            except Exception:
                print("Exception occured in Thread-4")
                print("Again running a thread-4")
                # t4.start()
                continue
            if kill == True:
                print("Thread-4 killed due to you.")
                break
            if showWindowVari == 1:
               cv2.imshow('eyes', EimgView)
               cv2.imshow("image", thresh)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


time.sleep(3)

t1 = threading.Thread(target=headPoseStarting)
t2 = threading.Thread(target=personDetectStarting)
t3 = threading.Thread(target=mouthDetectStarting)
t4 = threading.Thread(target=eyeDetector)
t5 = threading.Thread(target=openForm)

def startThreads():
    print("thread started")
#     t1.start()
#     t2.start()
#     t3.start()
#     t4.start()
#     joinThreads()

def joinThreads():
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()


t5.start()
time.sleep(20)
print("Sleep Done..")
t1.start()
t2.start()
t3.start()
t4.start()
joinThreads()

print("Main Ends here !")
cap.release()
cv2.destroyAllWindows()