import cv2, time

first_frame = None

video = cv2.VideoCapture(0)

while True:
    check, frame1 = video.read()
    check, frame2 = video.read()

    delta_frame = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(delta_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    thresh_frame = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 

    (_, cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1000: 
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Color Frame", frame1)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows