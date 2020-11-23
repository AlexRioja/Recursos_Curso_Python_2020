import utils
import imutils
import os
from cv2 import cv2
import numpy as np
import pickle
from imutils.video import VideoStream
from imutils.video import FPS
"""
    1.-Cargar los paths, el detector y el embeder 
"""
"""
Las siguientes lineas cargar el archivo de etiquetas y el recognizer(modelo que entrenamos en el paso anterior)
"""
with open("resources/pickle/labels.pickle", "rb") as f:
    inv_labels = pickle.load(f)
    labels = {v: k for k, v in inv_labels.items()}
recognizer = pickle.loads(open(paths['recognizer_path'], "rb").read())
"""
A continuación abrimos la interfaz de video con la webcam (Si da problemas, sustitiur 0 por 1)
e iniciamos el contador de FPS (para tener una idea del rendimiento)
"""
video_interface = cv2.VideoCapture(0)
# contador aproximado de FPS
fps = FPS().start()
"""
Recogemos un frame desde la interfaz de video
"""
while True:
    ret, frame = video_interface.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    # construimos el blob desde la imagen
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300))
    """
        1.-Marcar el input del detector
        2.-Detectar las caras que haya en la imagen, comparar con nuestro umbral de confianza
        3.-Obtener el recorte de la cara a partir de la imagen
    """
            try:
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                """
                    1.-Decirle al modelo de embedder el input(cara)
                    2.-Obtener el vector abstracto que representa esa cara (vec = embedder.forward())
                """
                embedder.setInput(faceBlob)
                vec = embedder.forward()
            except:
                print("Cara no centrada en el campo de vision de cámara. Frame corrupto")
            
            """
            Hasta aquí ya tenemos el vector que corresponde a la cara que estoy viendo, necesito por lo tanto introducirlo
            en la red para que me dé una respuesta de a quién pertenece la cara (array de predicciones)
            Esto se consigue pasandole el vector que tenemos a la red con el método predict_proba, tal y como
            se muestra a continuación.
            """
            preds = recognizer.predict_proba(vec)[0]
            print(preds)
            face_with_the_most_probability= np.argmax(preds)
            proba = preds[face_with_the_most_probability]
            """
            En la variable name tenemos el nombre de la cara con mayor probabilidad a ser correcta
            """
            name = labels[face_with_the_most_probability]
            print(name)

            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, startY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
    fps.update()
    """
    Lo unico que nos quedaría sería sacar una ventana para ver el reconocimiento a tiempo real
    utils-->show_face_detected(frame, face, text)
    Podemos informar en text que para salir hay que presionar la tecla 'Q'
    """

    """
    Gestionamos los eventos (q para salir)
    """
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
fps.stop()
print("FPS aproximados: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
video_interface.release()
print("Saliendo..")