import numpy as np
import cv2
# import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from helper import plotMatches
from opts import get_opts


# Q2.1.4


def matchPics(I1, I2, opts):
    """
    Match features across images

    Input
    -----
    I1, I2: Source images
    opts: Command line args

    Returns
    -------
    matches: List of indices of matched features across I1, I2 [p x 2]
    locs1, locs2: Pixel coordinates of matches [N x 2]
    """

    ratio = opts.ratio  # 'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  # 'threshold for corner detection using FAST feature detector'

    # TODO: Convert Images to GrayScale
    if len(I1.shape) > 2 and I1.shape[2] == 3:
        I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

    if len(I2.shape) > 2 and I2.shape[2] == 3:
        I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # TODO: Detect Features in Both Images
    fastI1 = corner_detection(I1, sigma)
    fastI2 = corner_detection(I2, sigma)

    # TODO: Obtain descriptors for the computed feature locations
    descI1, locs1 = computeBrief(I1, fastI1)
    descI2, locs2 = computeBrief(I2, fastI2)

    # TODO: Match features using the descriptors
    matches = briefMatch(descI1, descI2, ratio)

    return matches, locs1, locs2


if __name__ == "__main__":
    opts = get_opts()
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')
    matches, locs1, locs2 = matchPics(image1, image2, opts)
    plotMatches(image1, image2, matches, locs1, locs2)
