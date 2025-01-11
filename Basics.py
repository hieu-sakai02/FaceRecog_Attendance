import cv2
import numpy as np
import face_recognition

# STEP 1: LOADING IMAGES AND CONVERTING THEM TO RGB
# For Train Image
imgNimaPa = face_recognition.load_image_file('ImagesBasic/Stelle.jpg')
imgNimaPa = cv2.cvtColor(imgNimaPa,cv2.COLOR_BGR2RGB)
# For Test Image
imgTest = face_recognition.load_image_file('ImagesBasic/Stelle Test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# STEP 2: FINDING FACE LOCATIONS AND ENCODING FACE
# For Train Image
faceLoc = face_recognition.face_locations(imgNimaPa)[0]
encodeNimaPa = face_recognition.face_encodings(imgNimaPa)[0]
# For Test Image
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]

cv2.rectangle(imgNimaPa,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# STEP 3: COMPARING FACES AND FINDING DISTANCE BETWEEN THEM
results = face_recognition.compare_faces([encodeNimaPa],encodeTest)
faceDis = face_recognition.face_distance([encodeNimaPa],encodeTest)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('NimaPa',imgNimaPa)
cv2.imshow('NimaPaTest',imgTest)
cv2.waitKey(0)