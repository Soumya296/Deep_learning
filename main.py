import tensorflow as tf
import numpy as np
import cv2
import h5py
from tensorflow import keras
lst1 = np.range(0,24)
lst2 = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
dct = {zip(lst1,lst2)}

model = keras.models.load_model('My_model.h5')

cap = cv2.VideoCapture(0)

img_counter = 1 

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if not ret:
        print('Error in capturing frame, try again..')
        break
    cv2.imshow("Test", frame)

    key = cv2.waitKey(1)&0xFF

    if key == 27:
    
        print('Escape pressed, exiting the window')
        break

    elif key == ord('s'):
        
        #gray = cv2.cvtcolor(frame, cv2.COLOR_BGR2GRAY)

        img_name = "OpenCV_frame_{}.png".format(img_counter)
        #cv2.imwrirte(img_name, gray)
        cv2.imwrite(img_name, frame)
        print('{}written...'.format(img_name))


        img = cv2.imread(img_name,0)
        img_resized = cv2.resize(img, (28,28))
        img_reshaped = img_resized.reshape(1,28,28,1)

        img_test = np.array(img_reshaped, dtype = 'float64')
        img_test = img_test/255

        print('\n The letter is\n'+ str(dct[np.argmax(model.predict(img_test))]))

        img_counter += 1

cap.release()
cv2.destroyAllWindows()

        
