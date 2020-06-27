import numpy as np

def conv2d(I, K):
    # To-do
    # 1. 이미지와 커널의 합성곱을 한 후, 그 결과에 해당하는 크기를 가진 빈 행렬 만들기.
    ih, iw = I.shape
    kh, kw = K.shape
    
    rh, rw = ih-kh+1, iw-kw+1
    
    result = np.zeros((rh,rw))
    # 2. 컨볼루션 연산을 위한 반복문.
    for y in range(rh):
        for x in range(rw):
             # 3. I의 부분행렬(Kh * Kw)과 K의 합성곱을 한 후, 
            t = I[y:y+kh, x:x+kw] * K
             # 모두 더한 값을 새로운 행렬로 만들기.
            result[y,x] = np.sum(t)
      
    return result

    
print(conv2d(np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1]]), np.array([[1, -1], [1, -1]])))
