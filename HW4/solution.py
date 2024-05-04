import numpy as np
import cv2
import math
import random


def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START
    largest_set = []
    for i in range(10): #10번 반복
        sample_index = random.randint(0, len(matched_pairs))
        sample = matched_pairs[sample_index]
        
        orientation = (keypoints1[sample[0]][3] - keypoints2[sample[1]][3]) # 랜덤 샘플의 orientation
        scale = keypoints2[sample[1]][2] / keypoints1[sample[0]][2] # 랜덤 샘플의 scale
    
        inlier = [] # 인라이어에 속하는 매치 저장할 공간
        for j in range(len(matched_pairs)): # 모든 매치 쌍에 대하여
            orientation2 = (keypoints1[matched_pairs[j][0]][3] - keypoints2[matched_pairs[j][1]][3]) # 방향2
            scale2 = keypoints2[matched_pairs[j][1]][2] / keypoints1[matched_pairs[j][0]][2] # 스케일2
            # orientation2이 threshold 안이고 scale2도 threshold 안이라면 inlier로 판단
            if orientation2 > (orientation - math.pi*(orient_agreement/180)) and orientation2 < (orientation + math.pi*(orient_agreement/180)):
                if scale2 > scale * (1-scale_agreement) and scale2 < scale * (1+scale_agreement):
                    inlier.append(matched_pairs[j])

        if(len(inlier) > len(largest_set)): # 가장 많은 인라이어를 포함하는 랜덤 샘플
            largest_set = inlier[:] # deep copy
    ## END
    assert isinstance(largest_set, list)
    return largest_set



def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START
    ## the following is just a placeholder to show you the output format
    matched_pairs = []
    diff = [[] for _ in range(len(descriptors2))]  # descriptors1[i]과 descriptors2 모든 descriptor와의 차이를 저장
    for i in range(len(descriptors1)):
        for j in range(len(descriptors2)):
            diff[j] = [math.acos(np.dot(descriptors1[i], descriptors2[j])), j] # 정렬 후에 인덱스 정보를 확인하기 위해 인덱스를 같이 저장
        diff.sort(key = lambda x : x[0])
        if diff[0][0]/diff[1][0] < threshold:
            matched_pairs.append([i, diff[0][1]])
            
    ## END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START
    print(xy_points)
    print()
    one = [1]*len(xy_points)
    # 차원을 추가할 모든 요소가 1인 column을 xy_points의 길이만큼 생성한다.
    h_points = np.c_[xy_points, one]
    # xy_points에 one column을 추가하여 3차원을 추가한다.
    # 그 후h_points변수에 저장한다.

    xy_point_out = h.dot(h_points.T).T
    # h matrix에 h_points들을 내적하고 shape를 맞춰준다.
    nomal = xy_point_out[:, 2]
    # 3차원의 값들을 저장하여 값이 0이면 1e-10으로 수정한다.
    for i in nomal:
        if i == 0:
            i = 1e-10
    nomalizer = np.c_[nomal, nomal]
    # 나눠주는 수 nomalizer의 shape을 맞춰준다.
    xy_points_out = xy_point_out[:, [0,1]] / nomalizer
    # z값으로 x,y값을 나눠준다.
    print(xy_points_out)
    # END
    return xy_points_out

def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0

    # START



    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac