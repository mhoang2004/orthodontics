import numpy as np
import dlib
import cv2


def load_face_models(weight_path='./Stage1/DetectFace/ckpts/shape_predictor_68_face_landmarks.dat'):
    """Load dlib face detector and shape predictor once (CPU-only).

    Returns
    -------
    face_detector : dlib.fhog_object_detector
    shape_predictor : dlib.shape_predictor
    """
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(weight_path)
    return face_detector, shape_predictor


def face_landmark_detect(img, weight_path='./Stage1/DetectFace/ckpts/shape_predictor_68_face_landmarks.dat',
                         face_detector=None, shape_predictor=None):
    """Detect face landmarks.

    Parameters
    ----------
    img : numpy.ndarray
        Input image (BGR).
    weight_path : str
        Path to dlib shape predictor weights.  Ignored when *face_detector*
        and *shape_predictor* are supplied.
    face_detector : dlib.fhog_object_detector, optional
        Pre-loaded detector (avoids re-loading per call).
    shape_predictor : dlib.shape_predictor, optional
        Pre-loaded predictor (avoids re-loading per call).
    """
    if face_detector is None or shape_predictor is None:
        face_detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(weight_path)

    faces = face_detector(img, 0)
    try:
        shape = shape_predictor(img, faces[0])
    except:
        return None, None
    landmarks = np.array([(v.x, v.y) for v in shape.parts()])
    return faces[0], landmarks


def DetectFace(img_path, newsize=(512, 512), face_detector=None, shape_predictor=None):
    """Detect and crop face from image.

    Parameters
    ----------
    img_path : str
        Path to input image.
    newsize : tuple
        Target size for the cropped face.
    face_detector : dlib.fhog_object_detector, optional
        Pre-loaded dlib detector.
    shape_predictor : dlib.shape_predictor, optional
        Pre-loaded dlib shape predictor.
    """
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
        
    face, landmarks = face_landmark_detect(img, face_detector=face_detector,
                                           shape_predictor=shape_predictor)

    # make sure the coordinates will not exceed the img
    h, w = img.shape[:2]
    y1 = face.top() if face.top() > 0 else 0
    y1 = y1 if y1 < h else h-1
    y2 = face.bottom() if face.bottom() > 0 else 0
    y2 = y2 if y2 < h else h-1
    x1 = face.left() if face.left() > 0 else 0
    x1 = x1 if x1 < w else w-1
    x2 = face.right() if face.right() > 0 else 0
    x2 = x2 if x2 < w else w-1

    Face = img[y1:y2+1, x1:x2+1]
    Face = cv2.resize(Face, newsize)
    
    info = {
        'coord_x': (x1, x2+1),
        'coord_y': (y1, y2+1),
        'face_size': (x2+1-x1, y2+1-y1),
        'new_size': newsize,
    }

    return img, Face, info

if __name__ == '__main__':
    DetectFace(r'C:\IDEA_Lab\Project_tooth_photo\TeethSegm\Data\good/0a31887d774f4d439a87179045255951.jpg')