import dlib

cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
win = dlib.image_window()

img = dlib.load_rgb_image("heart.jpg")
dets = cnn_face_detector(img, 1)
rects = dlib.rectangles()

print("Numbers of faces detected: {}".format(len(dets)))
for index, det in enumerate(dets):
    r = det.rect    
    print("Detection {}: LEFT: {}, TOP: {}, RIGHT: {}, BOTTOM: {} : Confidence: {}".format(index, r.left(), r.top(), r.right(), r.bottom(), det.confidence ))
    rects.append(r)

win.clear_overlay()
win.set_image(img)
win.add_overlay(rects)
dlib.hit_enter_to_continue()
