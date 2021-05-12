import numpy as np
from skimage import filters, feature, img_as_int, transform
from math import pi, ceil
from scipy.spatial import KDTree


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
    # blur
    rescale = False
    if image.shape[0] < 1000 and image.shape[1] < 1000:
        rescale = True
        image = np.float32(transform.rescale(image, 2))
    sigma = 2.0
    alpha = 0.04
    blur_image = filters.gaussian(image, sigma=sigma)
    Iy, Ix = np.gradient(blur_image)
    Ixx = filters.gaussian(Ix ** 2, sigma=sigma)
    Iyy = filters.gaussian(Iy ** 2, sigma=sigma)
    Ixy = filters.gaussian(Ix * Iy, sigma=sigma)
    C = Ixx * Iyy - Ixy ** 2 - alpha * ((Ixx + Iyy) ** 2)
    C = C / np.linalg.norm(C)
    threshold = np.percentile(C, [60.0])
    np.putmask(C, C < threshold, 0)
    interested_points = feature.peak_local_max(
        C,
        min_distance=feature_width // 2,
        threshold_abs=2e-6,
        exclude_border=True,
        num_peaks=700,
    )
    if rescale:
        interested_points = interested_points / 2
    return interested_points[:, 1], interested_points[:, 0]


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
    if image.shape[0] < 1000 and image.shape[1] < 1000:
        image = np.float32(transform.rescale(image, 2))
        x = x * 2
        y = y * 2
    features = np.zeros((len(x), 128))
    # 获得4个重要的值
    grad_x = filters.sobel_v(image)
    grad_y = filters.sobel_h(image)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    grad_ori = np.arctan2(grad_y, grad_x)
    # 确定步长
    stride = feature_width // 4
    for j in range(len(x)):
        i = 0
        # print(ceil(x[j]), ceil(y[j]))
        # 开始按顺序求梯度直方图
        for x_ in range(ceil(x[j]) - 2 * stride, ceil(x[j]) + stride + 1, stride):
            for y_ in range(ceil(y[j]) - 2 * stride, ceil(y[j]) + stride + 1, stride):
                if grad_mag[y_:y_ + stride, x_:x_ + stride].shape != (4, 4):
                    continue
                s = get_bin(grad_mag[y_:y_ + stride, x_:x_ + stride],
                            grad_ori[y_:y_ + stride, x_:x_ + stride])
                features[j, 8 * i:8 * (i + 1)] = s
                i += 1
        feature = features[j] ** 0.6
        feature_norm = feature / np.linalg.norm(feature)
        threshold = np.percentile(feature_norm, [60.0])
        np.putmask(feature_norm, feature_norm < threshold, 0)
        feature_norm = feature_norm ** 0.7
        feature = feature_norm / np.linalg.norm(feature_norm)
        features[j] = feature
    return features


def PCA(features1, features2, m):
    C = (features1 - np.mean(features1, axis=0)) / (np.std(features1, axis=0) + 1e-5)
    cov_matrix = np.cov(C.T)
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
    i = np.argsort(-1 * np.abs(eigenvals))
    eigenvecs = eigenvecs[i]
    pca_features1 = np.dot(eigenvecs.T, features1.T)
    pca_features2 = np.dot(eigenvecs.T, features2.T)
    return pca_features1[:m].T, pca_features2[:m].T


def get_bin(grad_mag, grad_ori):
    s = np.zeros((1, 8))
    ori_list = np.arange(- np.pi, np.pi, np.pi / 4)
    inds = np.digitize(grad_ori, ori_list)
    for inds_i in range(8):
        mask = np.array(inds == inds_i)
        s[:, inds_i] = np.sum(grad_mag[mask])
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

    im1_features, im2_features = PCA(im1_features, im2_features, 36)

    ratio = 0.8
    matches = None
    confidences = None
    N1 = im1_features.shape[0]
    N2 = im2_features.shape[0]
    # im1_features = n * [fn1, fn2, fn3, ....., fnn]
    NND = np.zeros((N1, N2))
    # 求一个点到所有点的距离
    for i in range(N1):
        vec = im2_features - im1_features[i]
        NND[i] = np.linalg.norm(vec, axis=1)
    for i in range(im1_features.shape[0]):
        L = np.argsort(NND[i])
        min_index = L[0]
        second_min_index = L[1]
        rel = NND[i][min_index] / (NND[i][second_min_index] + 1e-8)
        confidence = 1 - rel
        if 0 < rel < ratio:
            m = np.array([i, min_index]).reshape((1, 2))
            c = np.array([confidence])
            if matches is None:
                matches = m
            else:
                matches = np.r_[matches, m]
            if confidences is None:
                confidences = c
            else:
                confidences = np.r_[confidences, c]
    return matches, confidences


def match_features_kdtree(im1_features, im2_features):

    im1_features, im2_features = PCA(im1_features, im2_features, 36)

    ratio = 0.8
    matches = None
    confidences = None
    tree = KDTree(im2_features)
    for i in range(im1_features.shape[0]):
        NN1, NN2 = tree.query(im1_features[i], k=2)
        min_dis, sec_dis = NN1
        min_index, _ = NN2
        rel = min_dis / (sec_dis + 1e-8)
        confidence = 1 - rel
        if 0 < rel < ratio:
            m = np.array([i, min_index]).reshape((1, 2))
            c = np.array([confidence])
            if matches is None:
                matches = m
            else:
                matches = np.r_[matches, m]
            if confidences is None:
                confidences = c
            else:
                confidences = np.r_[confidences, c]
    return matches, confidences
