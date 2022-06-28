import cv2
from tracker import *


def tracker(video, i_start, i_stop, j_start, j_stop):

    tracker = EuclideanDistTracker()
    cap = cv2.VideoCapture(video)

    # detectarea obiectului
    object_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

    i = 0
    j = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        height, width, _ = frame.shape

        # extrag regiunea de interes
        region_of_interest = frame[i_start: i_stop, j_start: j_stop]

        # detectarea obiectului
        mask = object_detector.apply(region_of_interest)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        _, otsu = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        mask = cv2.bilateralFilter(otsu, 9, 75, 75)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:

                x, y, w, h = cv2.boundingRect(contour)
                #cv2.rectangle(region_of_interest, (x, y), (x + w, y + h), (0, 255, 0), 3)

                detections.append([x, y, w, h])

        # 2. Incercarea de a face tracking

        k = 0

        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            k += 1

            cv2.rectangle(region_of_interest, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow("roi", region_of_interest)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        #if i % 30 == 0:
            #cv2.imwrite('Frame' + str(j) + '.jpg', frame)
            #f = open("videos/kpis/video3/file.txt", "a")
            #f.write("Frame: " + str(j) + ", detected no. of objects: " + str(k) + "\n")
            #f.close()
            #j += 1

        #i += 1


        key = cv2.waitKey(30)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    # functia tracker are nevoie de path-ul videoului
    # si de coordonatele regiunii de interes

    #tracker("videos/video0.mp4", 350, 720, 200, 900)
    #tracker("videos/video1.mp4", 350, 720, 100, 1100)
    tracker("videos/video2.mp4", 300, 720, 300, 1100)



