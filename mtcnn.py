from mtcnn.mtcnn import MTCNN
import cv2
import pprint

image = cv2.cvtColor(cv2.imread("bean.jpg"), cv2.COLOR_BGR2RGB)
detector = MTCNN()
result = detector.detect_faces(image)

bounding_box = result[0]['box']
keypoints = result[0]['keypoints']

pprint.pprint(result)

cv2.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2)

cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

cv2.imwrite("bean_drawn.jpg", image)
