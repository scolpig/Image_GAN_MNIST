import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

detector = dlib.get_frontal_face_detector()
shape = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

img = dlib.load_rgb_image('./imgs/02.jpg')
plt.figure(figsize=(16, 10))
plt.imshow(img)
plt.show()

img_result = img.copy()
dets = detector(img, 1)
print(len(dets))
fig, ax = plt.subplots(1, figsize=(10, 16))
for det in dets:

    x, y, w, h = det.left(), det.top(), det.width(), det.height()
    rect = patches.Rectangle((x, y), w, h,
                             linewidth=2, edgecolor='b', facecolor='None')
    ax.add_patch(rect)
ax.imshow(img_result)
plt.show()

fig, ax = plt.subplots(1, figsize=(16, 10))
obj = dlib.full_object_detections()

for detection in dets:
    s = shape(img, detection)
    obj.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y),
                                radius=3, edgecolor='b', facecolor='b')
        ax.add_patch(circle)
    ax.imshow(img_result)
plt.show()





