import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from vidgear.gears import CamGear

video_path = '3.mp4'

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (1020, 600))

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
 #   count += 1
 #   if count % 6 != 0:
 #       continue

    frame = cv2.resize(frame, (1020, 600))
    print(frame.shape)
    bbox, label, conf = cv.detect_common_objects(frame)

    person_bbox = [bbox[i] for i in range(len(bbox)) if label[i] == 'person']
    person_label = ['person'] * len(person_bbox)
    person_conf = [conf[i] for i in range(len(conf)) if label[i] == 'person']

    frame = draw_bbox(frame, person_bbox, person_label, person_conf)
    c = label.count('person')
    text = "Personas: " + str(c)
    cv2.putText(frame, text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    print(frame.shape)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()