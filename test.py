import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

faceCascade = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
font = cv2.FONT_HERSHEY_DUPLEX

kisi_id = 0
isimler = ['None','Melike','Emre']
camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)
minW = 0.1 * camera.get(3)
minH = 0.1 * camera.get(4)

while True:

    ret, img = camera.read()
    if not ret:
        continue
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yuzler = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in yuzler:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        kisi_id, dogruluk = recognizer.predict(gray[y:y + h, x:x + w])
        if(dogruluk > 70):
            kisi_id = isimler[kisi_id]
        else:
            kisi_id = "unknown"

        #Yazdırma
        cv2.putText(img, str(kisi_id), (x + 5, y - 5), font, 1, (0, 0, 255), 3)
        cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
print("\n terminated.")
camera.release()
cv2.destroyAllWindows()