'''
Inversion counting
n개의 숫자의 리스트 A가 주어질 때, inversion은 다음과 같이 정의된다.

만약 i < j 이고, A[i] > A[j]라면 A[i]와 A[j]는 inversion 관계이다.

예를 들어, A = [1, 4, 3, 2] 일 경우, 총 3개의 inversion이 존재하는데, 이는 그 값들을 나열해보면 (4, 3), (4, 2), (3, 2) 이다.

n개의 숫자가 주어질 때, inversion 관계인 숫자 쌍의 개수를 출력하는 프로그램을 작성하시오.
'''

import sys

def inversionCountInternal(data, start, end) :
    '''
    data[start] ~ data[end] 까지의 inversion 개수를 반환하고, 동시에 data[start] ~ data[end]를 정렬하는 함수
    '''
    if end - start <= 0 :
        return 0

    mid = (start + end) // 2

	# left, right 각각 계산해서 Inversion Count 반환 
    left = inversionCountInternal(data, start, mid)
    right = inversionCountInternal(data, mid+1, end)

    temp = []
	
    # left, right를 합치는 부분 
    leftPtr = start
    rightPtr = mid+1

    result = 0
    rightCnt = 0
    carryCnt = 0

    while leftPtr <= mid or rightPtr <= end :
        leftValue = data[leftPtr] if leftPtr <= mid else 987987987987987
        rightValue = data[rightPtr] if rightPtr <= end else 987987987987987

		# right가 크거나 같으면 temp에 left값을 넣고 left Ptr를 증가 
        # 왼쪽에서 빠지면 carryCnt는 변함없음 
        # rightCnt : 오른쪽에서 이제까지 빠진 갯수 
        if leftValue <= rightValue :
            if len(temp) >= 1:
                rightCnt += carryCnt
                carryCnt = 0

			# 왼쪽이 더 큰경우에 이제까지 더해준 rightCnt가 합해짐 
            result += rightCnt
            leftPtr += 1
            temp.append(leftValue)
            
        # left value가 right value보다 크면, CarryCnt를 계속 더해줌 
        # 오른쪽에서 빠지면 carryCnt를 1만큼씩 늘려줌 
        else :
            if len(temp) >= 1:
                rightCnt += carryCnt
                carryCnt = 1
            else :
                carryCnt += 1

            rightPtr += 1
            temp.append(rightValue)

    for i in range(start, end+1) :
        data[i] = temp[i-start]

    return left + right + result

def inversionCount(data) :
    '''
    n개의 숫자가 list로 주어질 때, inversion 관계에 있는 숫자 쌍의 개수를 반환하는 함수를 작성하세요.
    '''

    dataPrime = list(data)
    return inversionCountInternal(dataPrime, 0, len(data)-1)

def main():
    '''
    이 부분은 수정하지 마세요.
    '''

    data = [int(x) for x in input().split()]

    print(inversionCount(data))

if __name__ == "__main__":
    main()
