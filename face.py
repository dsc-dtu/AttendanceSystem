import cv2
import numpy as np
import io
dataset_path = './data/'
skip=0
face_data=[]
cap = cv2.VideoCapture(0)
file_name=input("Enter the name of person")
while True:
    ret,frame =cap.read()
    if ret==False:
        continue
    """Detects faces in the file located in Google Cloud Storage or the web."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    cv2.imwrite("/home/chirag/attendance/frame.jpg", frame) 
    #from PIL import Image, ImageDraw
    path="/home/chirag/attendance/frame.jpg"
    with io.open(path,'rb')  as image_file:
        content =image_file.read()
    image = vision.types.Image(content=content)
    #image.source.image_uri = uri
    #print(type(image))
    response = client.face_detection(image=image)
    faces = response.face_annotations
    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    #print('Faces:')
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
        
        
        face_section = frame[y_i:y_f,x_i:x_f]
        face_section = cv2.resize(face_section,(100,100))
        #frame[y_i,x_i]=[0,0,255]
        #frame[y_f,x_f]=[0,0,255]
        #cv2.rectangle(frame,(b[3].x,b[3].y),(b[1].x,b[1].y),(0,255,255),2)
        #print(b[3].y)
     
        face_data.append(face_section)
        print(len(face_data))

    cv2.imshow("Frame_s",face_section)
    cv2.imshow("Frame",frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('x'):
        break
face_data = np.asarray(face_data)   
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape) 
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully save at "+dataset_path+file_name+'.npy')  
        #print('face bounds: {}'.format(','.join(vertices)))
#detect_faces_uri("gs://attendance-system-237212-vcm/attendance1000.jpg")
cap.release()
cv2.destroyAllWindows()
