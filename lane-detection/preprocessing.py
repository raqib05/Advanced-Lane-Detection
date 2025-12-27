import numpy as np
import cv2
from matplotlib import pyplot as plt

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
    # for p in src:
    #     cv2.circle(img, tuple(p.astype(int)), 6, (0, 0, 255), -1)

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

    inverse_t = cv2.getPerspectiveTransform(dst, src)
    # cv2.imwrite("../data/coords.png", img)
    # cv2.imshow("test", img)
    # cv2.imshow("birds eye", birds_eye)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return birds_eye, t_matrix, inverse_t

def edge_detection(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 30, 60)
    # cv2.imshow("birds eye", edges)
    # cv2.waitKey(0)

    birds_eye = cv2.imread(img_path)
    hls = cv2.cvtColor(birds_eye, cv2.COLOR_BGR2HLS)
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    _, l_binary = cv2.threshold(l, 200, 255, cv2.THRESH_BINARY)
    _, s_binary = cv2.threshold(s, 120, 255, cv2.THRESH_BINARY)
    combined = np.zeros_like(edges)
    combined[(edges > 0) | (s_binary > 0)] = 255
    cv2.imwrite("../data/edges.png", combined)
    # cv2.imshow("birds eye", combined)
    # cv2.waitKey(0)

def sliding_window(img_path):
    binary = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert binary is not None, "Could not load image"

    H, W = binary.shape
    out_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Histogram (bottom half)
    histogram = np.sum(binary[H//2:, :], axis=0)

    midpoint = W // 2
    leftx_base  = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding window parameters
    nwindows = 9
    window_height = H // nwindows
    margin = 50
    minpix = 50

    # Identify nonzero pixels
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    # Sliding window loop
    for window in range(nwindows):
        win_y_low  = H - (window + 1) * window_height
        win_y_high = H - window * window_height

        win_xleft_low  = max(leftx_current - margin, 0)
        win_xleft_high = min(leftx_current + margin, W)

        win_xright_low  = max(rightx_current - margin, 0)
        win_xright_high = min(rightx_current + margin, W)

        cv2.rectangle(
            out_img,
            (win_xleft_low, win_y_low),
            (win_xleft_high, win_y_high),
            (0, 255, 255),
            2
        )

        cv2.rectangle(
            out_img,
            (win_xright_low, win_y_low),
            (win_xright_high, win_y_high),
            (0, 255, 255),
            2
        )
        good_left_inds = (
            (nonzeroy >= win_y_low) &
            (nonzeroy <  win_y_high) &
            (nonzerox >= win_xleft_low) &
            (nonzerox <  win_xleft_high)
        )

        good_right_inds = (
            (nonzeroy >= win_y_low) &
            (nonzeroy <  win_y_high) &
            (nonzerox >= win_xright_low) &
            (nonzerox <  win_xright_high)
        )

        left_inds = good_left_inds.nonzero()[0]
        right_inds = good_right_inds.nonzero()[0]

        left_lane_inds.append(left_inds)
        right_lane_inds.append(right_inds)

        # Recenter windows if enough pixels found
        if len(left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[left_inds]))

        if len(right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[right_inds]))

    # Concatenate pixel indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract lane pixel coordinates
    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def polynomial_fit(t_matrix):
    leftx, lefty, rightx, righty, out_img = sliding_window("../data/edges.png")
    lp = np.polyfit(lefty, leftx, 2)
    rp = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])

    left_fitx = lp[0]*ploty**2 + lp[1]*ploty + lp[2]
    right_fitx = rp[0]*ploty**2 + rp[1]*ploty + rp[2]
    
    left_coords = np.vstack([left_fitx, ploty])
    right_coords = np.vstack([right_fitx, ploty])

    ptsL = np.array([np.transpose(left_coords)], dtype=np.int32)
    ptsR = np.array([np.transpose(right_coords)], dtype=np.int32)

    cv2.polylines(out_img, [ptsL], isClosed=False, color=(255,0,0), thickness=3, lineType=cv2.LINE_AA)
    cv2.polylines(out_img, [ptsR], isClosed=False, color=(255,0,0), thickness=3, lineType=cv2.LINE_AA)

    cv2.imwrite("../data/polynomial_fit.png", out_img)

    img_size = (out_img.shape[1], out_img.shape[0])
    reverse_warp = cv2.warpPerspective(out_img, t_matrix, img_size)
    cv2.imshow("lanes", reverse_warp)
    cv2.waitKey(0)

    

_,_, t_matrix = perspective_warp("../data/lane5.png")
edge_detection("../data/birds_eye.png")
polynomial_fit(t_matrix)
leftx, lefty, rightx, righty, ans = sliding_window("../data/edges.png")

# cv2.imshow("sliding window", ans)
# cv2.waitKey(0)

# perspective_warp("../data/lane3.png")