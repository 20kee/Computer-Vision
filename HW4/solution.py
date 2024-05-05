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
    ones = np.ones(len(xy_points))
    d3_points = np.c_[xy_points, ones]

    new_d3_points = h.dot(d3_points.T).T

    for i in range(len(new_d3_points)):
        if new_d3_points[i][-1] == 0:
            new_d3_points[i][-1] = 1e-10
        new_d3_points[i] = new_d3_points[i] / new_d3_points[i][-1]

    xy_points_out = new_d3_points[:, [0,1]]
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
    max_inlier = 0 # 가장 많은 inlier의 수
    h = np.zeros(shape=(3, 3))
    for i in range(num_iter): # num_iter만큼 반복
        srcs = random.sample(range(len(xy_src)), 4) # homography matrix를 구할 4개의 좌표

        A = []
        for j in range(4): # 방정식을 풀기 위한 A 매트릭스를 만든다.
            A.append([xy_src[srcs[j]][0], xy_src[srcs[j]][1], 1, 0, 0, 0, -xy_ref[srcs[j]][0]*xy_src[srcs[j]][0], -xy_ref[srcs[j]][0]*xy_src[srcs[j]][1], -xy_ref[srcs[j]][0]])
            A.append([0, 0, 0, xy_src[srcs[j]][0], xy_src[srcs[j]][1], 1, -xy_ref[srcs[j]][1]*xy_src[srcs[j]][0], -xy_ref[srcs[j]][1]*xy_src[srcs[j]][1], -xy_ref[srcs[j]][1]])
        
        A = np.array(A)
        ATA = np.dot(A.T, A) # ATA 매트릭스를 구한다.
        v = np.linalg.svd(ATA, full_matrices=True)[2] # v에는 eigenvalue가 큰 순으로 eigenvector가 정렬되어 있다.
        h2 = np.reshape(v[-1], (3,3)) # 가장 작은 eigenvalue를 가지는 eigenvector가 homography matrix가 된다.

        inliers = 0 # 이제 inlier 개수를 구할 차례
        xy_proj = KeypointProjection(xy_src, h2) # 하나의 샘플로 구한 매트릭스로 다 프로젝션 시켜보자
        for k in range(len(xy_proj)):
            d = ( (xy_proj[k][0]-xy_ref[k][0])**2 + (xy_proj[k][1]-xy_ref[k][1])**2 ) ** (0.5) # 프로젝션 후 거리 차이
            if d < tol: # 거리가 threshold보다 작으면 inlier라고 판단
                inliers += 1

        if inliers > max_inlier: # inlier개수가 max_inlier보다 많으면 그때 matrix를 h에 저장
            max_inlier = inliers
            h = h2

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
