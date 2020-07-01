import sys


def getPowerSet(n, k):
    if n == k :
        return [[k]]
    
    result = [[k]]
    temp = []
    for i in range(k+1, n+1):
        temp = temp + getPowerSet(n,i)
        
    for i in range(len(temp)):
        temp[i] = [k] + temp[i]
        
    return result + temp
    
def powerSet(n) :
    '''
    n개의 원소를 가지는 집합 A의 멱집합의 원소를 사전 순서대로 list로 반환하는 함수를 작성하시오.

    예를 들어, n = 3 일 경우 다음의 list를 반환한다.

    [ [1], [1, 2], [1, 3], [1, 2, 3], [2], [2, 3], [3] ]
    '''
    result = []
    for i in range(1, n+1):
        result += getPowerSet(n, i)
    return result

def main():
    n = int(input())

    result = powerSet(n)
    
    for line in result :
        print(*line)

if __name__ == "__main__":
    main()

