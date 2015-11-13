# -*- coding: utf-8 -*-

__author__ = 'Tang Huan Song'

import os
import cv2
import numpy.linalg as linalg
import cv2.cv as cv
import numpy as np
import math
from functools import partial


# Script Parameters
# working codecs; but none of them seem to work for full-size videos
# 10 frames: 41,376kb
# CODEC = cv2.cv.CV_FOURCC('I', '4', '2', '0')
# 10 frames: 2,544kb
# CODEC = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
# 10 frames: 596kb
# CODEC = cv2.cv.CV_FOURCC('F', 'L', 'V', '1')
# 10 frames: 536kb
# CODEC = cv2.cv.CV_FOURCC('D', 'I', 'V', '3')
# 10 frames: 534kb
# CODEC = cv2.cv.CV_FOURCC('D', 'I', 'V', 'X')
# 10 frames: 528kb
CODEC = cv2.cv.CV_FOURCC('M', 'P', '4', '2')

# non-working
# 0kb output
# CODEC = cv2.cv.CV_FOURCC('A', 'Y', 'U', 'V')
# CODEC = cv2.cv.CV_FOURCC('I', 'U', 'Y', 'V')
# CODEC = cv2.cv.CV_FOURCC('Y', 'U', 'V', '1')

# encoder id 28 not found
# CODEC = cv2.cv.CV_FOURCC('A', 'V', 'C', '1')
# CODEC = cv2.cv.CV_FOURCC('H', '2', '6', '4')

# encoder id 21 not found
# CODEC = cv2.cv.CV_FOURCC('I', '2', '6', '3')

# throws resolution error: MPEG-1 does not support resolutions above 4095x4095
# CODEC = cv2.cv.CV_FOURCC('P', 'I', 'M', '1')

# throws resolution error:  H.263 does not support resolutions above 2048x1152
# CODEC = cv2.cv.CV_FOURCC('U', '2', '6', '3')


# CODEC = -1
HEURISTIC_VERTICAL_CROPPING = True
RESIZE_FACTOR = 0.5
WRITE_DEBUG_FRAME = True
DEBUG_FRAME_FILENAME = "DEBUG_FRAME.jpg"
OUTPUT_VIDEO_FILENAME = 'output.avi'
LEFT_VIDEO_FILENAME = 'media/football_left.mp4'
MIDDLE_VIDEO_FILENAME = 'media/football_mid.mp4'
RIGHT_VIDEO_FILENAME = 'media/football_right.mp4'


def make_synced_vid_iterator(left_camera_stream, middle_camera_stream, right_camera_stream):
    i = 1

    while i <= left_camera_stream.get(cv.CV_CAP_PROP_FRAME_COUNT) \
           and i <= middle_camera_stream.get(cv.CV_CAP_PROP_FRAME_COUNT)\
           and i <= right_camera_stream.get(cv.CV_CAP_PROP_FRAME_COUNT):
    # while i <= 10:
        _, left_frame = left_camera_stream.read()
        _, mid_frame = middle_camera_stream.read()
        _, right_frame = right_camera_stream.read()
        yield left_frame, mid_frame, right_frame, i
        i += 1


def filter_matches(matches, ratio=0.7):
    filtered_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            filtered_matches.append(m[0])

    return filtered_matches


def find_dimensions(image, homography):
    # initialize an array of [1,1,1]
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    # modify value at index 0 and 1 (excluding 2)
    (y, x) = image.shape[:2]

    base_p1[:2] = [0, 0]
    base_p2[:2] = [x, 0]
    base_p3[:2] = [0, y]
    base_p4[:2] = [x, y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:

        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

        hp_arr = np.array(hp, np.float32)

        normal_pt = np.array([hp_arr[0] / hp_arr[2], hp_arr[1] / hp_arr[2]], np.float32)

        if max_x is None or normal_pt[0, 0] > max_x:
            max_x = normal_pt[0, 0]

        if max_y is None or normal_pt[1, 0] > max_y:
            max_y = normal_pt[1, 0]

        if min_x is None or normal_pt[0, 0] < min_x:
            min_x = normal_pt[0, 0]

        if min_y is None or normal_pt[1, 0] < min_y:
            min_y = normal_pt[1, 0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return min_x, min_y, max_x, max_y


def stitch(base, next):
    detector = cv2.SURF()
    base_features, base_descriptors = detector.detectAndCompute(base, None)
    next_features, next_descriptors = detector.detectAndCompute(next, None)

    FLANN_INDEX_KDTREE = 1
    index_parameters = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_parameters = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_parameters, search_parameters)

    matches = matcher.knnMatch(next_descriptors, trainDescriptors=base_descriptors, k=2)

    matches_subset = filter_matches(matches)

    kp1 = []
    kp2 = []

    for match in matches_subset:
        kp1.append(base_features[match.trainIdx])
        kp2.append(next_features[match.queryIdx])

    p1 = np.array([k.pt for k in kp1])
    p2 = np.array([k.pt for k in kp2])

    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

    H = H / H[2, 2]
    H_inv = linalg.inv(H)

    min_x, min_y, max_x, max_y = find_dimensions(next, H_inv)

    # Adjust max_x and max_y by base img size
    max_x = max(max_x, base.shape[1])
    max_y = max(max_y, base.shape[0])

    move_h = np.matrix(np.identity(3), np.float32)
    if min_x < 0:
        move_h[0, 2] += -min_x
        max_x += -min_x

    if min_y < 0:
        move_h[1, 2] += -min_y
        max_y += -min_y
    move_x = move_h[0, 2]
    move_y = move_h[1, 2]
    # print "Homography: \n", H
    # print "Inverse Homography: \n", H_inv
    # print "Min Points: ", (min_x, min_y)

    mod_inv_h = move_h * H_inv

    img_w = int(math.ceil(max_x))
    img_h = int(math.ceil(max_y))

    # print "New Dimensions: ", (img_w, img_h)

    warped_img = warp_images(base, next, img_h, img_w, mod_inv_h, move_h)

    # Crop off the black edges
    warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(warped_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    best_rect = (0, 0, 0, 0)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # print "Bounding Rectangle: ", (x,y,w,h)

        deltaHeight = h - y
        deltaWidth = w - x

        area = deltaHeight * deltaWidth

        if area > max_area and deltaHeight > 0 and deltaWidth > 0:
            max_area = area
            best_rect = (x, y, w, h)
    image = crop_image(best_rect, warped_img, max_area)

    return dict(image=image, img_h=img_h, img_w=img_w, mod_inv_h=mod_inv_h,
                move_h=move_h, move_x=move_x, move_y=move_y, best_rect=best_rect, max_area=max_area)


def crop_image(best_rect, image, max_area):
    if max_area > 0:
        final_img_crop = image[best_rect[1]:best_rect[1] + best_rect[3], best_rect[0]:best_rect[0] + best_rect[2]]
        image = final_img_crop
    return image


def warp_images(base_image, next_image, img_h, img_w, mod_inv_h, move_h):
    # Warp the new image given the homography from the old image
    base_img_warp = cv2.warpPerspective(base_image, move_h, (img_w, img_h))
    next_img_warp = cv2.warpPerspective(next_image, mod_inv_h, (img_w, img_h))
    # Put the base image on an enlarged palette
    enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)
    # Create a mask from the warped image for constructing masked composite
    (ret, data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY),
                                    0, 255, cv2.THRESH_BINARY)
    enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp,
                                mask=np.bitwise_not(data_map),
                                dtype=cv2.CV_8U)
    # Now add the warped image
    final_img = cv2.add(enlarged_base_img, next_img_warp,
                        dtype=cv2.CV_8U)
    return final_img


def process_frame(left, mid, right, index, final=None, final_size=None, image_height=None, intermediate=None,
                  original_translation=None, total_frames=None):
    print "Processing frame: " + str(index) + "/" + str(total_frames) + "...",
    intermediate_img = warp_images(mid, left, intermediate['img_h'], intermediate['img_w'],
                                   intermediate['mod_inv_h'],
                                   intermediate['move_h'])
    intermediate_img = crop_image(intermediate['best_rect'], intermediate_img, intermediate['max_area'])
    final_img = warp_images(intermediate_img, right, final['img_h'], final['img_w'], final['mod_inv_h'],
                            final['move_h'])
    final_img = crop_image(final['best_rect'], final_img, final['max_area'])
    # remove top borders
    if HEURISTIC_VERTICAL_CROPPING is True:
        final_img = final_img[original_translation: original_translation + image_height]
    final_img = cv2.resize(final_img, final_size)
    return final_img


def main():
    os.chdir(os.getcwd())
    # specify the sources
    source = dict()
    source['left'] = cv2.VideoCapture(LEFT_VIDEO_FILENAME)
    source['mid'] = cv2.VideoCapture(MIDDLE_VIDEO_FILENAME)
    source['right'] = cv2.VideoCapture(RIGHT_VIDEO_FILENAME)

    synced_vid_iterator = make_synced_vid_iterator(source['left'], source['mid'], source['right'])

    # We need to calculate the transforms that we will be applying to the first frame.
    # Because the cameras don't move, we can reuse those transforms for the subsequent frames
    left, mid, right, index = synced_vid_iterator.next()
    print "Calculating per-frame transforms and processing frame " + str(index) + "...",

    # Calculate transform data from each stitch
    # Data for each stitch is stored into 'intermediate' and 'final' as dict values
    intermediate = stitch(mid, left)
    final = stitch(intermediate['image'], right)

    # Calculate vertical translation of middle image
    original_translation = intermediate['move_y'] + final['move_y']
    image_height = source['left'].get(cv.CV_CAP_PROP_FRAME_HEIGHT)

    final_img = final['image']

    # Crop image to be between the range of the middle image; this heuristic works well for this video set
    if HEURISTIC_VERTICAL_CROPPING is True:
        final_img = final_img[original_translation: original_translation + image_height]

    # Get final size to be used for resizing (if at all)
    final_size = tuple((np.array(final_img.shape)[:2] * RESIZE_FACTOR).astype(int)[::-1])

    # Resize to make smaller since I cannot yet find a codec that works with the full video
    final_img = cv2.resize(final_img, final_size)

    if WRITE_DEBUG_FRAME is True:
        cv2.imwrite(DEBUG_FRAME_FILENAME, final_img)
    print "Done."

    print "Initializing video writer...",
    if os.path.isfile(OUTPUT_VIDEO_FILENAME):
        os.remove(OUTPUT_VIDEO_FILENAME)
    out = cv2.VideoWriter(OUTPUT_VIDEO_FILENAME, CODEC, source['left'].get(cv.CV_CAP_PROP_FPS), final_size)
    print "Done."

    print "Beginning video writing."
    print "Writing frame " + str(index) + "...",
    out.write(final_img)
    print "Done."
    print "Applying transforms to subsequent frames."
    total_frames = int(source['left'].get(cv.CV_CAP_PROP_FRAME_COUNT))
    simplified_process_frame = partial(process_frame, final=final, final_size=final_size, image_height=image_height,
                                       intermediate=intermediate, original_translation=original_translation,
                                       total_frames=total_frames)

    for left, mid, right, index in synced_vid_iterator:
        final_img = simplified_process_frame(left, mid, right, index)
        print "Writing frame " + str(index) + "...",
        out.write(final_img)
        print "Done."

    out.release()


if __name__ == "__main__":
    main()
