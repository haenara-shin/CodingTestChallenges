import sys

def getSubsum(data) :
    '''
    n개의 숫자가 list로 주어질 때, 그 연속 부분 최대합을 반환하는 함수를 작성하세요.
    '''
    sum_num = []
    for i in range(len(data)):
        for j in range(i,len(data)):
            sum_num.append(sum(data[i:j]))

    return max(sum_num)

def main():
    data = [int(x) for x in input().split()]

    print(getSubsum(data))

if __name__ == "__main__":
    main()


