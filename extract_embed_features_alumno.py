import utils
import imutils
import os
from cv2 import cv2
import numpy as np
import pickle
"""
    1.-Cargar los paths, el detector, el embeder y recorrer el sistema de archivos(hecho)
"""
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "_").lower()
            print("Procesando: " + file + " con etiqueta : " + label)
            image = cv2.imread(path)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]
            imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                              swapRB=False, crop=False)

            """
            1.-Marcar el input del detector
            2.-Detectar las caras que haya en la imagen, comparar con nuestro umbral de confianza
            3.-Obtener el recorte de la cara a partir de la imagen
            4.-Mostrarlo en una ventana emergente (utils-->show_face_detected(frame, face_img, text))
            5.-Añadir una pequeña pausa para ver la imagen-->cv2.waitKey(50)#puede eliminarse para hacerlo más rápido
            """
            detector.setInput(imageBlob)
            faces = detector.forward()

            for i in range(0, faces.shape[2]):
                confidence = faces[0, 0, i, 2]
                if confidence > 0.5:
                    box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face = image[startY:endY, startX:endX]
                    utils.show_face_detected(image, face, "")
                    cv2.waitKey(50)

                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                         (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    """ hasta aquí tenemos la parte de la imagen correspondiente a la cara extraida, ahora:
                    1.-Decirle al modelo de embedder el input(cara)
                    2.-Obtener el vector abstracto que representa esa cara (vec = embedder.forward())
                    """
                    """
                    ¿Qué nos falta ahora? Recordemos que estamos realizando aprendizaje supervisado, donde tenemos las imágenes, 
                    y también tenemos etiquetas (labels) que dicen a quién pertenece cada imagen.

                    Nos interesaría tener por tanto alguna estructura del estilo:
                            {"Información de la cara": vec, "A quién pertenece la cara": label}
                    Que sería con lo que entrenaríamos a la red.

                    Si nos fijamos en el codigo que venía dado arriba, vemos que existe una variable "label", que se refiere
                    al nombre de la carpeta que contiene la imagen.

                    Por lo que, ya lo tenemos todo para poder crear la estructura anterior!, un úlitmo paso, las label ahora mismo
                    son 'str' y nos interesa que sean en formato numérico, ya que la red computa números, no palabras.
                    
                    Esto lo podemos conseguir con la función en utils-->create_ids_4_labels().
                    De la siguiente forma lo conseguimos:
                        1.-Crear una variable al inicio del programa llamada current_id con valor 0
                        2.-Crear las siguientes variables al inicio también:
                                labels = []
                                embeddings = []
                                label_ids = {} 
                        3.-label_int, current_id, labels_ids=utils.create_ids_4_labels(label, label_ids, current_id) 
                        4.-labels.append(label_int)
                        5.-embeddings.append(vec.flatten())
                    
                    Por lo tanto ya tendremos-->
                        label_ids={"Label String": label_str, "A qué label_int corresponde": label_int}
                        embeddings--> es un vector lleno de los datos asociados a cada cara (vec)
                        labels--> es un vector lleno de los ids numericos de las caras
                    Así que ya podemos construir la estructura de datos que buscabamos antes:
                        {"Información de la cara": embeddings, "A quién pertenece la cara": labels}
                    """
                    
"""
Llegados a este punto, ya hemos terminado de procesar todos los archivos, añadiremos las siguientes lineas
que guardan los datos anteriores para poder usarlos en el entrenamiento de la red
"""

data = {"embeddings": embeddings, "names": labels}
print("\nInformación de label_ids--> " +str(label_ids))
print("Información de labels--> " +str(labels))
#print("Información de nuestra estructura de datos--> " +str(data))
print("Se han procesado un total de: "+str(len(labels))+" caras!!!")
with open(paths['embed_path'], "wb") as f:
    f.write(pickle.dumps(data))

with open("resources/pickle/labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)
print("Saliendo..")