#Import necessary functions

import numpy as np
import cv2
from helper import loadVid
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
import multiprocessing


def ar_one_frame(i, frame, template, src_vidw, im1, opts):
    template = template[44:-44, src_vidw//2 - im1.shape[1]//2: src_vidw//2 + im1.shape[1]//2]
    template = cv2.resize(template, im1.shape[:2][::-1])
    
    matches, locs1, locs2 = matchPics(im1, frame, opts)
    locs1 = locs1[matches[:, 0], :]
    locs2 = locs2[matches[:, 1], :]

    locs1[:, [0, 1]] = locs1[:, [1, 0]]
    locs2[:, [0, 1]] = locs2[:, [1, 0]]

    H, _ = computeH_ransac(locs1, locs2, opts)
    frame = compositeH(H, template, frame)
    frame = frame.astype(np.uint8)
    return i, frame


def ar(opts):
    im1 = cv2.imread("../data/cv_cover.jpg")
    src_vid = loadVid("../data/ar_source.mov")
    dest_vid = loadVid("../data/book.mov")

    # saved below to inspect how to remove black borders
    # cv2.imwrite("./results/arframe0.png", src_vid[0])

    diff = dest_vid.shape[0] - src_vid.shape[0]
    extra = np.zeros((diff, src_vid.shape[1], src_vid.shape[2], src_vid.shape[3]))
    for i in range(diff):
        extra[i, :, :, :] = src_vid[i, :, :, :]

    src_vid = np.concatenate((extra, src_vid), axis=0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        "./results/ar.mp4", 
        fourcc, 
        25,
        (dest_vid[0].shape[1], dest_vid[0].shape[0])
    )

    _, src_vidw, _ = src_vid[0].shape

    mp_inputs = [(
        i,
        dest_vid[i], 
        src_vid[i],
        src_vidw, 
        im1, 
        opts
    ) for i in range(dest_vid.shape[0])]

    ar_frames = np.zeros(dest_vid.shape, dtype=np.uint8)
    def handle_result(result):
        ar_frames[result[0]] = result[1]

    n_cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(n_cpus)
    
    for i in range(len(mp_inputs)):
        pool.apply_async(ar_one_frame, args=mp_inputs[i], callback=handle_result)

    pool.close()
    pool.join()

    for frame in ar_frames:
        writer.write(frame)

    writer.release()

#Write script for Q3.1
if __name__=="__main__":
    opts = get_opts()
    ar(opts)
    