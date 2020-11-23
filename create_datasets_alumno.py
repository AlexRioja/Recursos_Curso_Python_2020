import argparse
import os 

# argparse para traer el valor de los parametros de entrada
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--label", required=True,
                help="Etiqueta que poner a las caras recogidas por el script")
args = vars(ap.parse_args())

try:
    os.mkdir("resources/faces_2_recognize/" + args['label'])
except OSError as error:
    pass

"""
    1-Cargar paths
    2-Cargar detector
"""
video_interface = cv2.VideoCapture(0)
"""
    1.-Bucle 50 iteraciones
"""
    """Dentro del bucle
    """
    ret, frame = video_interface.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300))
    """
        1.-Marcar el input del detector
        2.-Detectar las caras que haya en la imagen, comparar con nuestro umbral de confianza
        3.-Cuando tengamos la cara, sacar una ventana con la cara en cuestión-->utils.show_face_detected(frame, face_img, text
        4.-cv2.waitKey(500) #Capturar una cara cada 500ms 
        5.-Nuestro path a la imagen será algo como--> "resources/faces_2_recognize/" + args['label'] + "/" + str(n) + ".jpg"
        5.-Guardar el frame con la cara dentro de la carpeta asociada a la persona correspondiente-->utils.save_img(img_name, frame)
    """
# Liberamos la interfaz de video y destruimos las ventanas creadas
print('Saliendo...')
video_interface.release()
cv2.destroyAllWindows()