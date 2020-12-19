import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
face_detector = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')

def generation(ID):
    count = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite("dataset/user." + str(ID) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 30:
            break

x = int(input("For how many people the data set will be created:  "))

for i in range(0,x):
   ID = input('\n User ID : ')
   print("\n look at the camera :)")
   generation(ID)

cam.release()
cv2.destroyAllWindows()