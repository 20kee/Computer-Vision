from PIL import Image
import math
import numpy as np
"""
Get and use the functions associated with gaussconvolve2d that you used in the last HW02.
"""
def gauss1d(sigma):
    size = math.ceil(6*sigma)
    if size%2==0:
        size += 1
    gauss_array = np.array([math.exp(-math.pow(x,2)/(2*math.pow(sigma,2))) for x in range(-(size//2),size//2 + 1)])
    gauss_array /= gauss_array.sum()
    return gauss_array

def gauss2d(sigma):
    gauss_array = gauss1d(sigma)
    res_array = np.outer(gauss_array,gauss_array)
    res_array /= res_array.sum()
    return res_array

def convolve2d(array,filter):
    a,b = array.shape
    res_array = np.zeros((a,b), dtype="f")
    array = array.astype(np.float32)
    filter = np.flip(filter)
    f = (len(filter))
    ps = (f-1)//2
    array = np.pad(array, ((ps,ps),(ps,ps)), 'constant', constant_values=0)
    for i in range(a):
        for j in range(b):
            res_array[i][j] = (array[i:i+f, j:j+f]*filter).sum()
    return res_array

def gaussconvolve2d(array,sigma):
    filter = gauss2d(sigma)
    new_array = convolve2d(array, filter)
    return new_array

def sobel_filters(img):
    #x_filter와 y_filter를 선언한다.
    x_filter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
    y_filter = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)
    #받아온 이미지에 각각 x_filter와 y_filter를 convolution한다.
    Ix_array = convolve2d(img,x_filter)
    Iy_array = convolve2d(img,y_filter)
    #np.hypot함수를 사용하여 gradient value값을 배열에 저장한다.
    G = np.hypot(Ix_array,Iy_array)
    #np.arctan2함수를 사용하여 픽셀별 theta값을 저장한다.
    theta = np.arctan2(Iy_array,Ix_array)
    #픽셀값 범위 밖의 값들을 처리해준다.
    G = G/G.max() * 255
    np.where(G>255,255,G)
    np.where(G<0,0,G)
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
    return (G, theta)
    
    

def non_max_suppression(G, theta):
    #높이와 너비를 H, W에 저장한다.
    H, W = G.shape
    res = np.zeros((H,W), dtype=np.int32)
    #radian값을 degree값으로 바꿔준다.
    degree = theta * 180.0 / np.pi
    #음수 각들을 180씩 더하여 양수값의 0,45,90,135도로 바꿔준다.
    degree[degree<0] += 180
    #모든 값들을 돌면서 pixel값이 가장 높은 부분만 추출한다.
    #앞의 과정들을 convolution이기 때문에 3*3filter를 통해 zero padding이
    #상하좌우 1줄씩 생성됐다. 이 부분은 제외하고 반복문을 돈다. 
    for i in range(1, H-1):
        for j in range(1, W-1):
            q = 255
            r = 255
            #0도
            if (0 <= degree[i,j] < 22.5) or (157.5 <= degree[i,j] <= 180):
                q = G[i, j+1]
                r = G[i, j-1]
            #45도
            elif (22.5 <= degree[i,j] < 67.5):
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            #90도
            elif (67.5 <= degree[i,j] < 112.5):
                q = G[i+1, j]
                r = G[i-1, j]
            #135도
            elif (112.5 <= degree[i,j] < 157.5):
                q = G[i-1, j-1]
                r = G[i+1, j+1]
            #최대값 찾아서 반환한다.
            if (G[i,j] >= q) and (G[i,j] >= r):
                res[i,j] = G[i,j]
            else:
                res[i,j] = 0
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """
    return res
    
    

def double_thresholding(img):
    #과제 pdf에 나와있는 식을 이용하여 threshold를 지정한다.
    diff = img.max()-img.min()
    T_high = img.min() + diff * 0.15
    T_low = img.min() + diff * 0.03

    res = np.zeros(img.shape, dtype=np.int32)
    #T_high보다 큰 값은 strong edge로 설정하고 T_high보단 작지만 T_low보다
    #큰 값은 weak edge, T_low보다 작은 값은 non-relevant로 설정한다.
    #그리고 그에 해당하는 값을 부여한다(strong=255, weak=80, non_relevant=0)
    sx, sy = np.where(img>T_high)
    lx, ly = np.where((img<=T_high) & (img>=T_low))
    res[sx, sy] = 255
    res[lx, ly] = 80
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """
    return res

def dfs(img, res, i, j, visited=[]):
    # 호출된 시점의 시작점 (i, j)은 최초 호출이 아닌 이상 
    # strong 과 연결된 weak 포인트이므로 res에 strong 값을 준다
    res[i, j] = 255

    # 이미 방문했음을 표시한다
    visited.append((i, j))

    # (i, j)에 연결된 8가지 방향을 모두 검사하여 weak 포인트가 있다면 재귀적으로 호출
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)

def hysteresis(img):
    #전체 이미지에서 strong edge에 해당하는 픽셀에서 dfs를 돌려 strong edge에서
    #이어지는 weak edge를 strong edge로 바꿔주고 그렇지 못한 edge는 0으로 변환한다.
    M, N = img.shape
    visited = []
    #모든 픽셀이 0인 상태에서 edge로 판단되는 경우들만 255값을 주어 edge를 추출한다.
    res = np.zeros((M,N))
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == 255):
                if (i, j) not in visited:
                    dfs(img, res, i, j, visited)
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
    return res
#불러온 이미지를 흑백처리 후 blurring부터 해준다. sigma값은 1.6으로 설정한다.
im = Image.open("iguana.bmp")
im = im.convert('L')
im_array = np.asarray(im)
res1_array = gaussconvolve2d(im_array, 1.6)
res1 = Image.fromarray(res1_array.astype('uint8'))
res1.save("res1.bmp")
#sobel filter로 얻은 gradient 값으로 이미지를 얻는다.
G, theta = sobel_filters(res1_array)
res2_array = G.astype("uint8")
res2 = Image.fromarray(res2_array)
res2.save("res2.bmp")
#gradient value와 theta값을 이용하여 이미지를 얻는다.
res3_array = non_max_suppression(G, theta)
res3 = Image.fromarray(res3_array.astype('uint8'))
res3.save("res3.bmp")
#duoble threshold를 적용하여 이미지를 얻는다.
res4_array = double_thresholding(res3_array)
res4 = Image.fromarray(res4_array.astype('uint8'))
res4.save("res4.bmp")
#hysteresis함수를 적용하여 이미지를 얻는다.
res5_array = hysteresis(res4_array)
res5 = Image.fromarray(res5_array.astype('uint8'))
res5.save("res5.bmp")