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
    # 시그마*6 보다 큰 가장 작은 홀수
    length = math.ceil(sigma*6) + (1 if math.ceil(sigma*6)%2 == 0 else 0) 

    # 0 ~ 1-length 를 x로 보고 가우시안 식에 대입.
    # 이 결과는 정규화되지 않음
    unnormalized_gauss1d_filter = np.array([ ( 1 / math.sqrt(2*math.pi*sigma*sigma) ) * 
                                            ( 1 / math.e ** ((i - length//2)**2 / (2*(sigma**2))) ) for i in range(length) ])
    
    # 원소들의 합이 1이 나오도록 정규화
    gauss1d_filter = unnormalized_gauss1d_filter / sum(unnormalized_gauss1d_filter)
    return gauss1d_filter

# print(gauss1d(0.3), end='\n')
# print(gauss1d(0.5), end='\n')
# print(gauss1d(1), end='\n')
# print(gauss1d(2), end='\n')

def gauss2d(sigma):
    # 1D 가우시안 필터 두개의 외적을 통해 2D 가우시안 필터 생성
    gauss1d_filter = gauss1d(sigma)
    gauss2d_filter = np.outer(gauss1d_filter, gauss1d_filter)

    # 정규화 된 gauss1d를 사용하여 따로 정규화하지 않아도 이미 정규화된 상태이므로 바로 리턴.
    return gauss2d_filter

# print(gauss2d(0.5), end='\n')
# print(gauss2d(1), end='\n')

def convolve2d(array, filter):
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

def gaussconvolve2d(array, sigma):
    # 2D 필터 생성
    filter = gauss2d(sigma)

    # 콘볼루션
    filtered_array = convolve2d(array, filter)
    return filtered_array

def part1_4():
    # 이미지 오픈
    img = Image.open('3a_lion.bmp').convert('L')
    # 이미지를 어레이 형태로 변환
    img_array = np.asarray(img)

    # 가우시안 필터 적용
    filtered_array = gaussconvolve2d(img_array, 3)

    # uint8 형태로 변환 뒤 img show
    filtered_img = Image.fromarray(filtered_array.astype(np.uint8))
    filtered_img.show()
    return filtered_array
# part1_4()


#low frequency image 얻어오는 함수
def part2_1(img_file):
    # 이미지 오픈
    img = Image.open(img_file)

    # colors에는 R의 어레이, G의 어레이, B의 어레이 총 3개의 2차원 배열이 들어간다.
    colors = img.split()

    # colors를 콘볼루션한 이미지를 담을 배열 선언
    rgb_imgs = []
    for i in range(3):
        rgb_imgs.append(Image.fromarray(gaussconvolve2d(np.asarray(colors[i]), 3).astype(np.uint8)))

    # merge 함수를 통해 RGB 세개의 배열을 다시 하나로 합쳐준다.
    filtered_img = Image.merge('RGB', ((rgb_imgs[0], rgb_imgs[1], rgb_imgs[2])))
    filtered_img.show()
    low_freq_array = np.asarray(filtered_img)
    return low_freq_array
# part2_1('3a_lion.bmp')

# high frequency image 얻어오는 함수
def part2_2(img_file):
    # 이미지 오픈
    img = Image.open(img_file)
    img_array = np.asarray(img)

    # part2_1 함수를 통해 low freq image 획득
    low_freq_array = part2_1(img_file)

    # 음수의 값을 갖는 수를 제거하기 위해 0~255의 평균인 128을 더해준다.
    high_freq_array = img_array - low_freq_array + 128
    
    high_freq_img = Image.fromarray(high_freq_array)
    high_freq_img.show()
    return high_freq_array

# part2_2('3b_tiger.bmp')

def part2_3():
    # 각 이미지에 대해 low freq img와 high freq img를 얻는다.
    low_freq_array = part2_1('3a_lion.bmp')
    high_freq_array = part2_2('3b_tiger.bmp')

    # 연산에서 255보다 큰 값이 발생하므로 uint16타입으로 변환한 후 더하여야 한다.
    new_array = low_freq_array.astype(np.int16) + high_freq_array.astype(np.int16)
    
    # 0~255의 값으로 변환
    new_array =  new_array / np.amax(new_array) * 255

    # 이미지로 읽을 때는 다시 uint8 타입으로 변경해주어야 한다.
    new_img = Image.fromarray(new_array.astype(np.uint8))
    new_img.show()
    return new_array

part2_3()