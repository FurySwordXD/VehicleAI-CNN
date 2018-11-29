import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
import os

def keys_to_output(keys):
    # [W, A, S, D]
    output = [0, 0, 0, 0]

    if 'A' in keys:
        output[1] = 1
    elif 'D' in keys:
        output[3] = 1
    elif 'W' in keys:
        output[0] = 1
    elif 'S' in keys:
        output[2] = 1
    return output


file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previously saved training data...')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, creating new training data...')
    training_data = [] 

for i in list(range(5))[::-1]:
    print(i+1)
    time.sleep(1)

last_time = time.time()
paused = False
while(True):

    keys = key_check()

    if not paused:
        screen = np.array(grab_screen(region=(0,40,800,600)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (80,60))

        output = keys_to_output(keys)
        training_data.append([screen, output])

        if len(training_data) % 500 == 0:
            print('Writing to file!')           
            np.save(file_name, training_data)
            print('Completed!')
            print(len(training_data))
        #newscreen, original_image, m1, m2 = process_img(screen)

        print('FPS: %d Time: %.2f' %( 1/(time.time() - last_time), time.time() - last_time))
        last_time = time.time()

        
        #cv2.imshow('window2', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        #cv2.imshow('window', newscreen)

        #if cv2.waitKey(25) & 0xFF == ord('x'):
        #    cv2.destroyAllWindows()
        #    break

    if 'P' in keys:
        if paused:
            paused = False
            time.sleep(1)
        else:
            paused = True
            time.sleep(1)