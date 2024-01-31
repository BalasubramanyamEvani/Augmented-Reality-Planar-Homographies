import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

# Import necessary functions

# Q2.2.4

def warpImage(opts):
    print(f"opts max_iters: {opts.max_iters}, inlier_tol: {opts.inlier_tol}")
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')
    image3 = cv2.imread("../data/hp_cover.jpg")

    image3 = cv2.resize(image3, (image1.shape[1], image1.shape[0]))
    
    matches, locs1, locs2 = matchPics(image1, image2, opts)
    locs1 = locs1[matches[:, 0], :]
    locs2 = locs2[matches[:, 1], :]

    locs1[:, [0, 1]] = locs1[:, [1, 0]]
    locs2[:, [0, 1]] = locs2[:, [1, 0]]
    
    H, _ = computeH_ransac(locs1, locs2, opts)
    composite_img = compositeH(H, image3, image2)
    
    cv2.imshow("warped", composite_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    opts = get_opts()
    warpImage(opts)
