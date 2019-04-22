import cv2
import numpy as np
dataset_path = './data/'
skip=0
face_data=[]
cap = cv2.VideoCapture(0)
file_name=input("Enter the name of person")
def detect_faces_uri(uri):
    """Detects faces in the file located in Google Cloud Storage or the web."""
    from google.cloud import vision
    import cv2
    #from PIL import Image, ImageDraw
    frame=cv2.imread("/home/chirag/Desktop/attendance1000.jpg")
    client = vision.ImageAnnotatorClient()
    image = vision.types.Image()
    image.source.image_uri = uri
    print(type(image))
    response = client.face_detection(image=image)
    faces = response.face_annotations
    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    print('Faces:')
    offset=10
    for face in faces:
        #print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
        #print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        #print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))
        #vertices = (['({},{})'.format(vertex.x, vertex.y)
                    #for vertex in face.bounding_poly.vertices]:
                        #print(type(vertex))
        b =[]

        for vertex in face.bounding_poly.vertices:
            b.append(vertex)
        x_i=int(b[0].x)
        x_f=int(b[2].x)
        y_i=int(b[0].y)
        y_f=int(b[2].y)
        w=x_f-x_i
        
        face_section = frame[y_i:y_f,x_i:x_f]
        frame[y_i,x_i]=[0,0,255]
        frame[y_f,x_f]=[0,0,255]
        #cv2.rectangle(frame,(b[3].x,b[3].y),(b[1].x,b[1].y),(0,255,255),2)
        #print(b[3].y)
        cv2.imshow("Frame",face_section)
        key_pressed = cv2.waitKey(0) & 0xFF
        if key_pressed == ord('x'):
            continue
        
        #print('face bounds: {}'.format(','.join(vertices)))
detect_faces_uri("gs://attendance-system-237212-vcm/attendance1000.jpg")
cv2.destroyAllWindows()