import sys

def getRect(heights) :
    '''
    n개의 판자의 높이가 주어질 때, 이를 적당히 잘라 얻을 수 있는 직사각형의 최대 넓이를 반환하는 함수를 작성하세요.
    '''
    n = len(heights)
	
    # 재귀 호출 종료 조건 : 크기가 1 이하이면, 가로 1 x 세로인 넓이 출력 
    if n <= 1 :
        return heights[0]

	# Pivot : mid값 기록 
    mid = n//2
	
    # left, right 각각 재귀호출 진행 
    left = getRect(heights[:mid])
    right = getRect(heights[mid:])

	# Point 기록 
    leftPtr = mid
    rightPtr = mid

	# 중앙 histogram 넓이 계산 
    curHeight = min(heights[leftPtr], heights[rightPtr])
    result = curHeight * (rightPtr - leftPtr + 1)

	# 왼쪽과 오른쪽으로 Point를 하나씩 옮겨가면서 넓이 계산 
    # 종료 조건 : 왼쪽 오른쪽 모두 확장할 공간이 없을 시 종료 
    while leftPtr-1 >= 0 or rightPtr+1 < n :
    	# 왼쪽과 오른쪽 중에 height가 높은 곳으로 point를 이동 
        leftHeight = heights[leftPtr-1] if leftPtr-1 >= 0 else -987987987987987
        rightHeight = heights[rightPtr+1] if rightPtr+1 < n else -987987987987987
        maxHeight = max(leftHeight, rightHeight)
	
        if leftHeight >= rightHeight :
            leftPtr -= 1
        else :
            rightPtr += 1

        curHeight = min(curHeight, maxHeight)
        
        # result를 기록해서 가장 max값만 저장
        result = max(result, curHeight * (rightPtr - leftPtr + 1))

    return max([result, left, right])

def main():
    '''
    이 부분은 수정하지 마세요.
    '''

    data = [int(x) for x in input().split()]

    print(getRect(data))

if __name__ == "__main__":
    main()
