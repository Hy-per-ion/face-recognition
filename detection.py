import cv2


def boundary(image, classifier, scaleFactor, minNeighbour, color, text):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbour)
    co = []
    for (x, y, w, h) in features:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        co = [x, y, w, h]
    return co


def detect(image, faceCas, eyeCas, noseCas, mouthCas):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    co = boundary(img, faceCas, 1.1, 10, color['blue'], "Face")
    if len(co) == 4:
        roi_img = img[co[1]:co[1] + co[3], co[0]:co[0] + co[2]]
        co = boundary(roi_img, eyeCas, 1.1, 14, color['red'], "Eye")
        co = boundary(roi_img, noseCas, 1.1, 6, color['green'], "Nose")
        co = boundary(roi_img, mouthCas, 1.1, 20, color['white'], "Mouth")
    return image


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml ")
eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
noseCascade = cv2.CascadeClassifier('Nariz.xml')
mouthCascade = cv2.CascadeClassifier('Mouth.xml')
video_cap = cv2.VideoCapture(0)
while True:
    _, img = video_cap.read()
    img = detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade)
    cv2.imshow("face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()
