import sys

def binarySearch(data, m) :
    '''
    n개의 숫자 중에서 m이 존재하면 "Yes", 존재하지 않으면 "No"를 반환하는 함수를 작성하세요.
    '''
    low = 0
    high = len(data) - 1
    
    while high >= low:
        mid = (low + high) // 2
        if m < data[mid]:
            high = mid -1
        elif m == data[mid]:
            return 'Yes'
        else:
            low = mid + 1
    return 'No'
    

def main():
    '''
    이 부분은 수정하지 마세요.
    '''

    data = [int(x) for x in input().split()]
    m = int(input())

    print(binarySearch(data, m))

if __name__ == "__main__":
    main()

