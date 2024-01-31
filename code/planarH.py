import numpy as np
import random
import cv2


def computeH(x1, x2):
    #Q2.2.1
    #Compute the homography between two sets of points
    assert x1.shape == x2.shape
    total_points = x1.shape[0]
    A = np.zeros((2 * total_points, 9))
    for i in range(0, total_points):
        X1, Y1 = x1[i][0], x1[i][1]  
        X2, Y2 = x2[i][0], x2[i][1]
        A[2 * i] = np.array([
            -X2, -Y2, -1, 0, 0, 0, X1 * X2, X1 * Y2, X1
        ])
        A[2 * i + 1] = np.array([
            0, 0, 0, -X2, -Y2, -1, Y1 * X2, Y1 * Y2, Y1
        ])
    u, s, vt = np.linalg.svd(A)
    H2to1 = vt[-1].reshape((3, 3))
    return H2to1


def computeH_norm(x1, x2):
    #Q2.2.2
    #Compute the centroid of the points
    mewx1 = np.mean(x1, axis=0)
    mewx2 = np.mean(x2, axis=0)

    #Shift the origin of the points to the centroid
    x1shifted = x1 - mewx1
    x2shifted = x2 - mewx2

    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    max_euclidean_x1 = np.max(np.linalg.norm(x1shifted, ord=2, axis=1))
    max_euclidean_x2 = np.max(np.linalg.norm(x2shifted, ord=2, axis=1))

    scale1 = np.sqrt(2) / max_euclidean_x1
    scale2 = np.sqrt(2) / max_euclidean_x2
    
    x1shifted = scale1 * x1shifted
    x2shifted = scale2 * x2shifted
    
    #Similarity transform 1
    T1 = np.array([
        [scale1, 0, -scale1 * mewx1[0]],
        [0, scale1, -scale1 * mewx1[1]],
        [0, 0, 1]
    ])

    #Similarity transform 2
    T2 = np.array([
        [scale2, 0, -scale2 * mewx2[0]],
        [0, scale2, -scale2 * mewx2[1]],
        [0, 0, 1]
    ])

    #Compute homography
    H = computeH(x1shifted, x2shifted)

    #Denormalization
    H2to1 = np.linalg.inv(T1) @ (H @ T2)
    return H2to1


def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inliers

    assert locs1.shape[0] == locs2.shape[0]

    total_matched_points = locs1.shape[0]
    bestH2to1 = np.zeros((3, 3))
    
    locs1 = np.hstack((locs1, np.ones((total_matched_points, 1))))
    locs2 = np.hstack((locs2, np.ones((total_matched_points, 1))))
    inliers = 0
    
    all_indexes = range(0, total_matched_points)

    for i in range(max_iters):
        selected_indexes = random.sample(all_indexes, 4)
        leftImage = locs1[selected_indexes, :]
        rightImage = locs2[selected_indexes, :]
        
        H = computeH_norm(leftImage, rightImage)
        locs1hat = (H @ locs2.T).T
        locs1hat = locs1hat / (locs1hat[:, 2][:, np.newaxis] + 1e-10)
        
        t = np.linalg.norm(locs1hat - locs1, ord=2, axis=1)
        tmp_inliers = np.sum(t[t < inlier_tol])

        if tmp_inliers > inliers:
            bestH2to1 = np.copy(H)
            inliers = tmp_inliers

    return bestH2to1, inliers


def compositeH(H2to1, template, img):
    #Create a composite image after warping the template image on top
    #of the image using the homography
    #Note that the homography we compute is from the image to the template;
    #x_xtemplate = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    #Create mask of same size as template
    #Warp mask by appropriate homography
    #Warp template by appropriate homography
    #Use mask to combine the warped template and the image
    Hinv = np.linalg.inv(H2to1)
    composite_image = np.zeros(img.shape)
    composite_image = np.copy(img)
    mask = np.ones(template.shape, dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(
        mask, Hinv, (img.shape[1], img.shape[0])
    )
    warp_template = cv2.warpPerspective(
        template, Hinv, (img.shape[1], img.shape[0])
    )
    indexes = np.where(warped_mask == 255)
    composite_image[indexes] = warp_template[indexes]
    return composite_image
