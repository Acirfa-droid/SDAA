# Clasificador para Rpi basado en CNN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Funciones de preprocesado:
def image_preproc(img, coef = None, width = None, height = None, inter = cv2.INTER_AREA):
    dim = (width,height)
    # RGB to Gray image conversion
    gray = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
    # resize the image
    img_prep = cv2.resize(gray, dim, interpolation = inter)
    # rescale the image
    img_prep.astype('float32') # Convierte a float32
    img_prep = img_prep/coef # Escalado
    
    # return the resized image
    return img_prep

num_letters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
               6: 'G', 7: 'H', 8: 'I', 9: 'K',
               10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
               15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
               20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

model= Sequential()
model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
          
model.add(Flatten())

model.add(Dense(units=512,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=24,activation='softmax'))
model.summary()

model.load_weights('prueba3.h5')

#%% Predecimos una foto

# Prueba con una imagen:
from time import time
from picamera import PiCamera
import copy
from sense_hat import SenseHat
sense=SenseHat()

# Define la camara:
camera = PiCamera()
camera.resolution = (640,480)
camera.rotation = 180
camera.start_preview(fullscreen=False, window=(30,30,640,480)) 

print('Prueba de cámara (PULSAR INTRO)')
okay = input() # SE PULSA INTRO, ES ÚNICAMENTE PARA INICIAR LA CÁMARA POR PRIMERA VEZ
# Le llega una imagen:
output = np.empty((480, 640, 3), dtype=np.uint8)
camera.capture(output, 'rgb')
print('Prueba realizada!')

while(1):
    # Espera a que llegue una imagen:
    print('Esperando imagen (PULSAR INTRO CUANDO SE DESEE CAPTURAR IMAGEN)')
    okay = input() ## SE PULSA INTRO, ES ÚNICAMENTE PARA INICIAR LA CÁMARA PsOR PRIMERA VEZ
    # Le llega una imagen:
    output = np.empty((480, 640, 3), dtype=np.uint8)
    camera.capture(output, 'rgb')
    print('Imagen capturada')
    signal = copy.copy(output)
    
    # Empieza el proceso:
    start_time = time() # Tiempo de ejecución comienza

    # Preprocesado:
    ancho = 28
    alto = 28
    signal_prep = image_preproc(signal, coef = 255, width = ancho, height = alto)
    """test = signal_prep.reshape([-1,ancho, alto,1])"""
    plt.imshow(signal_prep)
    img_array = np.array(signal_prep)
    img_array = img_array[np.newaxis, :,:, np.newaxis]
    
    # Clasificación: 
    prediccion=num_letters[np.argmax(model.predict(img_array))]
    print("PREDICCIÓN:", num_letters[np.argmax(model.predict(img_array))])
    print("Porcentajes de prediccion: ", model.predict(img_array))
    


    # Termina el proceso:
    elapsed_time = time() - start_time # Tiempo de ejecución termina
    print("Tiempo empleado: %.10f seconds." % elapsed_time) # Imprime el tiempo que ha tardado

    
    print('¿Clasificacion correcta? (y/n)')
    select1 = input()
    if select1=="y":
        print("MARAVILLA")
        sense.show_letter(prediccion)
        time.sleep(1)
        sense.clear()

# Imprime la clase predicha y la imagen original:
try:
    camera.stop_preview()
    camera.close()
except:
    print("niguna imagen se ha clasificado correctamente")
# Cierra la vista previa y libera la camara:
    camera.stop_preview()
    camera.close()
