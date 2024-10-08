from sklearn.neighbors import KNeighborsClassifier
import cv2 as cv
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

def get_name_by_id(student_id):
    with open('data/student.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if row[0] == student_id:
                return row[1]
    return None

if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

video = cv.VideoCapture(0)
facedetect = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/ids.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ['ID', 'NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray_frame, 1.3 ,5)

    for (x, y, w, h) in faces:
        
        cropped_img = frame[y:y+h, x:x+w, :]
        resized_img = cv.resize(cropped_img, (50,50)).flatten().reshape(1,-1)
        output = knn.predict(resized_img)

        recognized_id = output[0]
        recognized_name = get_name_by_id(recognized_id)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M")

        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

        cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv.putText(frame, str("Press 'p' to record your attendance"), (40, 30), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(frame, str("Press 'q' to quit"), (40, 60), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(frame, str(output[0]), (x,y-15), cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        
        attendance = [str(recognized_id), str(recognized_name), str(timestamp)]

    cv.imshow("Smart Attendance System",frame)
    k = cv.waitKey(1)
    if k == ord('p'):
        speak("Attendance Recorded")
        time.sleep(2)
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
    if k == ord('q'):
        break
video.release()
cv.destroyAllWindows()