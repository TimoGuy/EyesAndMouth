# python3 -m pip install opencv-python

import cv2

def draw_boundary(img, classifiers, scaleFactor, minNeighbors, color, texts):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    count = 0
    for classifier in classifiers:
        features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors, minSize=(30, 30))
        coords = []
        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, texts[count], (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            coords.append([x, y, w, h])

        count += 1

    return coords, img


def detect(img, classifiers, texts):
    color = { "blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0) }
    coords, img = draw_boundary(img, classifiers, 1.1, 20, color['blue'], texts)
    return img


eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')
video_capture = cv2.VideoCapture(0)    # 0 is the default webcam. If in clamshell mode, the external webcam should be the default one still

while True:
    _, img = video_capture.read()
    # img = detect(img, [eye_cascade, mouth_cascade], ["eye", "mouth"])
    img = detect(img, [eye_cascade], ["eye"])
    cv2.imshow("face det", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()    
