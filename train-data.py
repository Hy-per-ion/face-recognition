import cv2


def generate_dataset(image, id, image_id):
    cv2.imwrite("data/user." + str(id) + "." + str(image_id) + ".jpg", image)


def boundary(image, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    co = []
    for (x, y, w, h) in features:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        co = [x, y, w, h]
    return co


def detect(image, faceCas, image_id):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    co = boundary(image, faceCas, 1.1, 10, color['blue'], "Face")
    if len(co) == 4:
        roi_img = image[co[1]:co[1] + co[3], co[0]:co[0] + co[2]]
        user_id = 1
        generate_dataset(roi_img, user_id, image_id)

    return image


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
img_id = 0

while True:
    if img_id % 50 == 0:
        print("Collected ", img_id, " images")
    _, img = video_capture.read()
    img = detect(img, faceCascade, img_id)
    cv2.imshow("face detection", img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
