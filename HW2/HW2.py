from PIL import Image
import numpy as np
import math

def boxfilter(n):
#짝수 예외처리
    assert n%2!=0, "dimension must be odd"
    #element가 1/(n*n)이고 크기가 n*n인 배열 생성
    box_filter = np.full((n,n), 1/(n*n))
    return box_filter
# print(boxfilter(3), end='\n')
# print(boxfilter(4), end='\n')
# print(boxfilter(7), end='\n')

def gauss1d(sigma):
    length = math.ceil(sigma*6)+ (1 if math.ceil(sigma*6)%2 == 0 else 0) # 시그마*6 보다 큰 가장 작은 홀수

    # 0 ~ 1-length 를 x로 보고 가우시안 식에 대입.
    # 이 결과는 정규화되지 않음
    unnormalized_gauss1d_filter = np.array([ ( 1 / math.sqrt(2*math.pi*sigma*sigma) ) * ( 1 / math.e ** ((i - length//2)**2 / (2*(sigma**2))) ) for i in range(length) ])
    
    # 원소들의 합이 1이 나오도록 정규화
    gauss1d_filter = unnormalized_gauss1d_filter / sum(unnormalized_gauss1d_filter)
    return gauss1d_filter

# print(gauss1d(0.3), end='\n')
# print(gauss1d(0.5), end='\n')
# print(gauss1d(1), end='\n')
# print(gauss1d(2), end='\n')

def gauss2d(sigma):
    # 1D 가우시안 필터 두개의 외적을 통해 2D 가우시안 필터 생성
    gauss2d_filter = np.outer(gauss1d(sigma), gauss1d(sigma))
    return gauss2d_filter

# print(gauss2d(0.5), end='\n')
# print(gauss2d(1), end='\n')

def convolve2d(array, filter):
    # 콘볼루션을 위한 필터 뒤집기
    flipped_filter = np.flip(filter)
    filter_length = filter.shape[1]

    filtered_array = np.zeros(array.shape)
    rows, columns = array.shape
    for n in range(rows):
        for m in range(columns):
            # 각 픽셀에 대해 필터 적용
            for n2 in range(n - filter_length//2, n + filter_length//2 + 1):
                for m2 in range(m - filter_length//2, m + filter_length//2 + 1):
                    # 필터와 곱할 픽셀 값을 얻어옴. 만약 Array의 범위를 벗어난다면 zero padding을 고려함.
                    pixel = array[n2][m2] if n2 >= 0 and n2 < rows and m2 >= 0 and m2 < columns else 0
                    filtered_array[n][m] += pixel * flipped_filter[n2-n+filter_length//2][m2-m+filter_length//2]

    return filtered_array

def gaussconvolve2d(array, sigma):
    # 2D 필터 생성
    filter = gauss2d(sigma)

    # 콘볼루션
    filtered_array = convolve2d(array, filter).astype(np.uint8)
    return filtered_array

def part1_4():
    img = Image.open('3a_lion.bmp')
    img_array = np.asarray(img)
    #implement
    # greyscale로 변환하여 저장할 array 선언
    img_array_greyscale = np.zeros((img_array.shape[0], img_array.shape[1]))
    for n in range(img_array.shape[0]):
        for m in range(img_array.shape[1]):
            # 변환 공식
            img_array_greyscale[n][m] = 0.2989 * img_array[n][m][0] + 0.5870 * img_array[n][m][1] + 0.1140 * img_array[n][m][2]
    
    # 가우시안 필터 적용
    filtered_array = gaussconvolve2d(img_array_greyscale, 3)
    filtered_img = Image.fromarray(filtered_array)
    filtered_img.show()
    return filtered_array
# part1_4()

def part2_1():
    img = Image.open('3a_lion.bmp')
    img_array = np.asarray(img)
    red_array = img_array[:, :, 0]
    filtered_red_array = gaussconvolve2d(red_array, 3)
    print(filtered_red_array)
    green_array = img_array[:, :, 1]
    filtered_green_array = gaussconvolve2d(green_array, 3)
    print(filtered_green_array)
    blue_array = img_array[:, :, 2]
    filtered_blue_array = gaussconvolve2d(blue_array, 3)
    print(filtered_blue_array)

    low_freq_list = []
    for n in range(img_array.shape[0]):
        temp_list = []
        for m in range(img_array.shape[1]):
            temp_list.append([filtered_red_array[n][m], filtered_green_array[n][m], filtered_blue_array[n][m]])
        low_freq_list.append(temp_list)
    low_freq_array = np.array(low_freq_list)


    filtered_img = Image.fromarray(low_freq_array.astype(np.uint8))
    filtered_img.show()
    return low_freq_array
# part2_1()

def part2_2():
    img = Image.open('3a_lion.bmp')
    img_array = np.asarray(img)
    low_freq_array = part2_1()

    high_freq_array = img_array - low_freq_array
    print(np.min(high_freq_array))
    high_freq_array = high_freq_array + np.min(high_freq_array)
    print(high_freq_array)


    high_freq_array2 = high_freq_array + np.min(high_freq_array) + 30
    print(high_freq_array2)

    high_freq_img = Image.fromarray(high_freq_array.astype(np.uint8))
    high_freq_img.show()


    high_freq_img = Image.fromarray(high_freq_array2.astype(np.uint8))
    high_freq_img.show()
    return high_freq_array

part2_2()