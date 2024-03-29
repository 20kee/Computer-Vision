from PIL import Image
import math
import numpy as np

"""
Get and use the functions associated with gaussconvolve2d that you used in the last HW02.
"""
def gauss1d(sigma):
        # 시그마*6 보다 큰 가장 작은 홀수
    length = math.ceil(sigma*6) + (1 if math.ceil(sigma*6)%2 == 0 else 0) 

    # 0 ~ 1-length 를 x로 보고 가우시안 식에 대입.
    # 이 결과는 정규화되지 않음
    unnormalized_gauss1d_filter = np.array([ ( 1 / math.sqrt(2*math.pi*sigma*sigma) ) * 
                                            ( 1 / math.e ** ((i - length//2)**2 / (2*(sigma**2))) ) for i in range(length) ])
    
    # 원소들의 합이 1이 나오도록 정규화
    gauss1d_filter = unnormalized_gauss1d_filter / sum(unnormalized_gauss1d_filter)
    return gauss1d_filter

def gauss2d(sigma):
    # 1D 가우시안 필터 두개의 외적을 통해 2D 가우시안 필터 생성
    gauss1d_filter = gauss1d(sigma)
    gauss2d_filter = np.outer(gauss1d_filter, gauss1d_filter)

    # 정규화 된 gauss1d를 사용하여 따로 정규화하지 않아도 이미 정규화된 상태이므로 바로 리턴.
    return gauss2d_filter

def convolve2d(array,filter):
    # 콘볼루션을 위한 필터 뒤집기
    flipped_filter = np.flip(filter)
    filter_length = filter.shape[1]

    rows, columns = array.shape

    # 모서리부분 콘볼루션을 위해 제로 패딩을 입혀준다.
    padding_size = (filter_length-1)//2
    array = np.pad(array, ((padding_size, padding_size),(padding_size, padding_size)), 'constant', constant_values=0)

    # 콘볼루션 결과를 저장할 array
    filtered_array = np.zeros((rows, columns))
    for row in range(rows):
        for column in range(columns):
            filtered_array[row][column] = (array[row:row+filter_length, column:column+filter_length]*flipped_filter).sum()

    return filtered_array

def gaussconvolve2d(array,sigma):
    # 2D 필터 생성
    filter = gauss2d(sigma)

    # 콘볼루션
    filtered_array = convolve2d(array, filter)
    return filtered_array

def reduce_noise(img):
    """ Return the gray scale gaussian filtered image with sigma=1.6
    Args:
        img: RGB image. Numpy array of shape (H, W, 3).
    Returns:
        res: gray scale gaussian filtered image (H, W).
    """
    greyscale_img = img.convert("L")
    greyscale_img_array = np.asarray(greyscale_img, np.float32)

    res = gaussconvolve2d(greyscale_img_array, 1.6)
    return res

def sobel_filters(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
    """
    x_filter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
    y_filter = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)

    x_intensity = convolve2d(img, x_filter)
    y_intensity = convolve2d(img, y_filter)

    G = np.hypot(x_intensity, y_intensity)
    theta = np.arctan2(x_intensity, y_intensity)

    G = G/G.max() * 255
    np.where(G>255,255,G)
    np.where(G<0,0,G)

    return (G, theta)

def non_max_suppression(G, theta):
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """
    pass
    return res

def double_thresholding(img):
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """
    #implement     
    return res

def dfs(img, res, i, j, visited=[]):
    # calling dfs on (i, j) coordinate imply that
    #   1. the (i, j) is strong edge
    #   2. the (i, j) is weak edge connected to a strong edge
    # In case 2, it meets the condition to be a strong edge
    # therefore, change the value of the (i, j) which is weak edge to 255 which is strong edge
    res[i, j] = 255

    # mark the visitation
    visited.append((i, j))

    # examine (i, j)'s 8 neighbors
    # call dfs recursively if there is a weak edge
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)

def hysteresis(img):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: hysteresised image.
    """
    #implement 

    return res

def main():
    RGB_img = Image.open('./iguana.bmp')

    noise_reduced_img = reduce_noise(RGB_img)
    Image.fromarray(noise_reduced_img.astype('uint8')).save('./iguana_blurred.bmp', 'BMP')
    
    g, theta = sobel_filters(noise_reduced_img)
    Image.fromarray(g.astype('uint8')).save('./iguana_sobel_gradient.bmp', 'BMP')
    Image.fromarray(theta.astype('uint8')).save('./iguana_sobel_theta.bmp', 'BMP')

    # non_max_suppression_img = non_max_suppression(g, theta)
    # Image.fromarray(non_max_suppression_img.astype('uint8')).save('./iguana_non_max_suppression.bmp', 'BMP')

    # double_threshold_img = double_thresholding(non_max_suppression_img)
    # Image.fromarray(double_threshold_img.astype('uint8')).save('./iguana_double_thresholding.bmp', 'BMP')

    # hysteresis_img = hysteresis(double_threshold_img)
    # Image.fromarray(hysteresis_img.astype('uint8')).save('./iguana_hysteresis.bmp', 'BMP')

main()