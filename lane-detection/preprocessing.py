import numpy as np
import cv2

def perspective_warp(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (960, 540))
    src = np.float32([
        (0.41, 0.65),  # top-left
        (0.19, 0.9),  # bottom-left
        (0.62, 0.65),  # top-right
        (0.95, 0.9)   # bottom-right
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
    cv2.imwrite("../data/birds_eye.png", birds_eye)
    # cv2.imwrite("../data/coords.png", img)
    # cv2.imshow("test", img)
    # cv2.imshow("birds eye", birds_eye)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return birds_eye

def edge_detection(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 30, 60)
    # cv2.imshow("birds eye", edges)
    # cv2.waitKey(0)


# perspective_warp("../data/lane5.png")
# edge_detection("../data/birds_eye.png")

# perspective_warp("../data/lane3.png")