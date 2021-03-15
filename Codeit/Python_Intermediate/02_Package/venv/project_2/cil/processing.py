# 이미지 색상 반전
def invert(img):
    height, width = len(img), len(img[0])
    new_img = empty_image(height, width)
    for i in range(height):
        for j in range(width):
            new_img[i][j] = invert_bit(img[i][j])
    return new_img


# 이미지 합성
def merge(img1, img2):
    # img1과 img2의 크기가 같은지 확인
    if get_size(img1) == get_size(img2): ### 코드를 작성해 주세요 ### :
        new_img = empty_image(get_size(img1)[0], get_size(img1)[1])
        for i in range(get_size(img1)[0]):
            for j in range(get_size(img1)[1]):
                if img1[i][j] == 1 and img2[i][j] == 1:
                    new_img[i][j] = 1
                elif img1[i][j] == 0 and img2[i][j] == 0:
                    new_img[i][j] = 0
                else:
                    new_img[i][j] = 1
        ### 코드를 작성해 주세요 ### 
        
        return new_img
    else:
        print('img1과 img2의 크기가 같아야 합니다!')


# 이미지 크기를 계산해 주는 함수
def get_size(img):
    return [len(img), len(img[0])]


# -1로 채워진 새로운 이미지 생성
def empty_image(height, width):
    new_img = []
    for i in range(height):
        new_img.append([-1] * width)
    return new_img


# 비트 반전
def invert_bit(bit):
    return 1 - bit


# 비트 합성
def or_bits(bit1, bit2):
    return min(1, bit1 + bit2)