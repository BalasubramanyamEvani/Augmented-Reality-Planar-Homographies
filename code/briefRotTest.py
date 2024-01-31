import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import scipy
import matplotlib.pyplot as plt
import skimage

# Q2.1.6

def plot(im1, im2, matches, locs1, locs2):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.axis('off')
    skimage.feature.plot_matches(
        ax, im1, im2, locs1, locs2, matches, matches_color='r', only_matches=True)
    plt.show()
    return

def rotTest(opts):
    # Read the image and convert to grayscale, if necessary
    img = cv2.imread("../data/cv_cover.jpg")
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    x = []
    y = []

    for i in range(36):
        angle = i * 10
        # Rotate Image
        rotated_img = scipy.ndimage.rotate(img, angle)

        # Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(img, rotated_img, opts)

        # visualize
        if i in [1, 10, 20]:
            plot(img, rotated_img, matches, locs1, locs2)

        # Update histogram
        x.append(i)
        y.append(len(matches))

        print(f"Iteration: {i}")

    # Display histogram
    plt.figure()
    plt.bar(x, y)
    plt.xlabel("rotation angle (x 10 degrees)")
    plt.ylabel("number of matches")
    plt.xticks(x, rotation=90)
    plt.title("Q2.1.6 - count of matches for each orientation")
    plt.show()


if __name__ == "__main__":
    opts = get_opts()
    rotTest(opts)
