import numpy as np
import cv2

def perspective_warp(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (960, 540))
    src = np.float32([
        (0.43, 0.67),  # top-left
        (0.15, 0.95),  # bottom-left
        (0.62, 0.67),  # top-right
        (0.95, 0.95)   # bottom-right
    ])
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src*img_size
    for p in src:
        cv2.circle(img, tuple(p.astype(int)), 6, (0, 0, 255), -1)

    img_size = (int(img_size[0][0]), int(img_size[0][1]))
    dst = np.float32([
        [0, 0],
        [0, img_size[1]],
        [img_size[0], 0],
        [img_size[0], img_size[1]]
    ])
    t_matrix = cv2.getPerspectiveTransform(src, dst)
    birds_eye = cv2.warpPerspective(img, t_matrix, img_size)
    cv2.imshow("test", img)
    cv2.imshow("birds eye", birds_eye)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  

perspective_warp("../data/lane3.png")