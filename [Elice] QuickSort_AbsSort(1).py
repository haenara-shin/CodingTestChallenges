import sys
sys.setrecursionlimit(100000)

def quickSort(array) :
    # 1. 기저조건 : array가 1개 이하이면 자기자신 출력 
    if len(array) < 2 :
        return array

    # 2. 임의로 pivot 출력 (중앙에 있는 값) 
    pivot = array[len(array)//2]

    # 3. left : pivot보다 작은 값, right : pivot보다 큰 값, equal : pivot하고 같은 값(절댓값 분리를 위함) 
    left = []
    right = []
    equal = []


    for x in array :
        # pivot이 크면 왼쪽 
        if abs(pivot) > abs(x) :
            left.append(x)

        # pivot이 작으면 오른쪽 
        elif abs(pivot) < abs(x) :
            right.append(x)

        # pivot이랑 같은데, 음수이면 왼쪽 양수이면 오른쪽 
        else :
            if x < 0 :
                equal = [x] + equal
            else :
                equal.append(x)

    # 4. 재귀호출로 left, right 각각에 대해 QuickSort진행
    return quickSort(left) + equal + quickSort(right)

def sortAbs(array):
    '''
    절댓값을 기준으로 오름차순 정렬한 결과를 반환하는 함수를 작성하세요.
    '''
    return quickSort(array)

def main():
    line = [int(x) for x in input().split()]

    print(*sortAbs(line))

if __name__ == "__main__":
    main()
