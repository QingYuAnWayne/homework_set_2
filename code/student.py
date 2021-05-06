import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from math import pi,ceil
from scipy.spatial.distance import cdist
import cv2


def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to contact TAs

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the project instructions

    # These are placeholders - replace with the coordinates of your interest points!
    xs = []
    ys = []
    return xs, ys


def get_features(image, x, y, feature_width):
    '''
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to contact TAs

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here! See block comments and the project instructions

    # This is a placeholder - replace this with your features!
    # 初始化
    features = np.zeros((len(x), 128))
    # 获得4个重要的值
    grad_x = filters.sobel_v(image)
    grad_y = filters.sobel_h(image)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_ori = np.arctan2(grad_y, grad_x)
    #确定步长
    stride = feature_width // 4
    for j in range(len(x)):
        i = 0
        # print(ceil(x[j]), ceil(y[j]))
        # 开始按顺序求梯度直方图
        for x_ in range(ceil(x[j]) - 2 * stride, ceil(x[j]) + stride + 1, stride):
            for y_ in range(ceil(y[j]) - 2 * stride, ceil(y[j]) + stride + 1, stride):
                s = get_bin(grad_mag[x_:x_ + stride, y_:y_ + stride],
                            grad_ori[x_:x_ + stride, y_:y_ + stride])
                features[j, 8 * i:8 * (i + 1)] = s
                i += 1
    return features


def get_bin(grad_mag, grad_ori):
    s = np.zeros((1, 8))
    for i in range(8):
        theta1 = pi * 2 / 8 * i - pi
        theta2 = pi * 2 / 8 * (i + 1) - pi
        # 处于该范围的就取出来相加
        tmp = np.where((theta1 <= grad_ori) & (grad_ori < theta2), True, False)
        s[:, i] = grad_mag[tmp].sum()
    return s


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the project instructions

    # These are placeholders - replace with your matches and confidences!
    ratio = 1
    matches = None
    confidences = None
    NND = cdist(im1_features, im2_features, 'euclidean')
    # 找到NND在im1中的所有点与im2中所有点的欧式距离中最小的那一个的index （针对每行的index）
    index_min = np.argmin(NND, axis=1)
    # 找到最小的欧式距离
    NN1 = np.min(NND, axis=1)
    # 找到最大的欧氏距离
    max_num = np.max(NND)
    # 把刚才的最小的点欧式距离变成最大的欧式距离+1
    for i in range(NND.shape[0]):
        NND[i, index_min[i]] = max_num + 1
    # 这里找到的就是第二小的欧式距离了
    NN2 = np.min(NND, axis=1)
    # 防止NN2 还是0， 所以加一个极其小的数， 得到的NN1/NN2
    NN1_NN2 = NN1 / (NN2 + 1e-4)
    for i in range(NN1_NN2.shape[0]):
        if 0 < NN1_NN2[i] < ratio:
            # match success
            index = index_min[i]
            p1 = i
            p2 = index
            m = np.array([p1, p2]).reshape((1, 2))
            c = np.array([1-NN1_NN2[i]])
            if matches is None:
                matches = m
            else:
                matches = np.r_[matches, m]
            if confidences is None:
                confidences = c
            else:
                confidences = np.r_[confidences, c]
    return matches, confidences
