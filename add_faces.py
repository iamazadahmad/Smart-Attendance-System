import cv2 as cv;
import numpy as np
import pickle
import os
import csv

video = cv.VideoCapture(0)

face_detect = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0

name = input("Enter your name: ").upper()
id = input("Enter your Id: ").upper()

def check_and_store_student(id, name):
    file_path = 'data/student.csv'

    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Name"])

    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if row[0] == id:
                print("Student already registered.")
                return False

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([id, name])

    print(f"Student {name} with ID {id} registered successfully.")
    return True

if not check_and_store_student(id, name):
    video.release()
    cv.destroyAllWindows()
    exit()

while True:
     ret, frame = video.read()
     gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
     faces = face_detect.detectMultiScale(gray_frame, 1.3, 5)

     for(x, y, w, h) in faces:
          cropped_img = frame[y:y+h, x:x+w, :]
          resized_img = cv.resize(cropped_img, (50, 50))
          if (len(faces_data) <= 100 and i % 5 == 0):
               faces_data.append(resized_img)
          i = i + 1
          cv.putText(frame, str(f'{name}, please rotate your neck while taking your facial data'), (40, 30), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
          cv.putText(frame, str(f'Taking your 100 facial data: {len(faces_data)} / 100'), (40, 60), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
          cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

     cv.imshow('Smart Attendance System', frame)
     k = cv.waitKey(1)
     if k == ord('q') or len(faces_data) == 100:
          break

video.release()
cv.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

if 'ids.pkl' not in os.listdir('data/'):
    ids = [id] * 100
    with open('data/ids.pkl', 'wb') as f:
        pickle.dump(ids, f)
else:
    with open('data/ids.pkl', 'rb') as f:
        ids = pickle.load(f)
    ids = ids + [id] * 100
    with open('data/ids.pkl', 'wb') as f:
        pickle.dump(ids, f)

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)