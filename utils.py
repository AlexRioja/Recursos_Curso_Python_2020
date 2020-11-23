from cv2 import cv2

def load_paths():
    """Returns a dictionary with all the paths loaded
    """
    dataset_path = "resources/faces_2_recognize"
    embed_path = "resources/pickle/embeddings.pickle"
    embed_model = "resources/model/openface_nn4.small2.v1.t7"
    model_path = "resources/model/res10_300x300_ssd_iter_140000.caffemodel"
    model_proto_path = "resources/model/deploy.prototxt"
    recognizer_path = "resources/pickle/recognizer.pickle"
    paths={"dataset_path":dataset_path, "embed_path":embed_path, "embed_model":embed_model,
            "model_path":model_path, "model_proto_path":model_proto_path, "recognizer_path":recognizer_path}
    return paths

def load_face_detector_and_embedder(model_path, model_proto_path, embed_model):
    """Returns loaded face detector and embedder 

    Args:
        model_path (str): Path to the detector model
        model_proto_path (str): Path to the proto.txt file from detector model
        embed_model (str): Path to the embedder model (OpenFace)
    """
    detector = cv2.dnn.readNetFromCaffe(model_proto_path, model_path)  # carga del detector de caras
    embedder = cv2.dnn.readNetFromTorch(embed_model)  # carga del modelo embedder
    return detector, embedder

def show_face_detected(frame, face_img, text):
	"""Creates and displays 2 windows, one with the feed from the webcam and the other with
		the detected face alone
	Args:
		frame (img): frame of the webcam input
		face_img (img): cropped image of the face
		text (str): text to show in the main window
	"""
	try:
		cv2.putText(frame, text, (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3,
		                (66, 53, 243), 2, cv2.LINE_AA)
		    
		cv2.imshow('Ventana principal', frame)
		cv2.imshow('Recorte con la cara detectada', face_img)
	except:
		pass

def save_img(img_name, frame):
    """Resizes the given image with diferent interpolation algorithms
       and then saves it with the given img_name
    Args:
        img_name (str): Path where the img is going to be stored
        frame (img): Image to store
    """
    if frame.shape < (300, 300):
        frame = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_AREA)
    else:
        frame = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(img_name+ ".jpg", frame)

def create_ids_4_labels(label_str, label_ids, current_id):
    """Assign numerical values to label_str variable

    Args:
        label_str (str): Label in string format
        label_ids (dict): Contains assotiation: {label_str, label_int}
        current_id (int): Incremental value

    Returns:
        label_int (int): Label in int format
        current_id (int): Incremented current_id
        label_ids (dict): Updated label_ids dictionary
    """
    label_int = 0
    if not label_str in label_ids:
        label_ids[label_str] = current_id
        current_id += 1
    label_int = label_ids[label_str]
    return label_int, current_id, label_ids
