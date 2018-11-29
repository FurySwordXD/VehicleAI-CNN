import numpy as np
import cv2
import time
from grabscreen import grab_screen
import os
from alexnet import alexnet
from keys import key_check, PressKey, ReleaseKey, W, A, S, D

t_time = 0.09

def forward():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def left():
    PressKey(A)
    PressKey(W)
    ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(A)

def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(D)

def backward():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(W)


WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCH = 8
MODEL_NAME = 'models/car-ai-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCH)
model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

print('Loading model %s...' % MODEL_NAME)
print('Starting in...')
for i in list(range(5))[::-1]:
    print(i+1)
    time.sleep(1)

last_time = time.time()
paused = False
while(True):

    if not paused:
        screen = np.array(grab_screen(region=(0,40,800,600)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (80,60))

        #newscreen, original_image, m1, m2 = process_img(screen)

        print('FPS: %d Time: %.2f' %( 1/(time.time() - last_time), time.time() - last_time))
        last_time = time.time()

        prediction = model.predict([screen.reshape(WIDTH, HEIGHT, 1)])[0]
        print(prediction)

        turn_thresh = .75
        fwd_thresh = 0.70

        if prediction[0] > fwd_thresh:
            forward()
        elif prediction[1] > turn_thresh:
            left()
        elif prediction[3] > turn_thresh:
            right()
        elif prediction[2] > fwd_thresh:
            backward()
        else:
            forward()

    keys = key_check()

    if 'P' in keys:
        if paused:
            paused = False
            time.sleep(1)
        else:
            paused = True
            ReleaseKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            time.sleep(1)
    elif 'X' in keys:
        break