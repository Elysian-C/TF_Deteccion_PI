import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

video_path = '1.mp4'

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (1020, 600))

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
#    count += 1
#    if count % 6 != 0:
#        continue

    frame = cv2.resize(frame, (1020, 600))
    print(frame.shape)
    bbox, label, conf = cv.detect_common_objects(frame, enable_gpu=True)

    thresh = 0.6

    f_bbox = [bbox[i] for i in range(len(bbox)) if label[i] in ['person'] and conf[i] > thresh]
    f_label = [label[i] for i in range(len(label)) if label[i] in ['person'] and conf[i] > thresh]
    f_conf = [conf[i] for i in range(len(conf)) if label[i] in ['person'] and conf[i] > thresh]

    frame = draw_bbox(frame, f_bbox, f_label, f_conf)
    p = label.count('person')
    cv2.putText(frame, "Personas: " + str(p), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
    cv2.imshow("Deteccion de personas", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()