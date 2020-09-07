def diffDigit(a, b) :
    '''
    a, b의 서로 다른 자리수의 개수를 반환한다
    '''

    a = str(a)
    b = str(b)

    len_a = len(a)
    len_b = len(b)
    cal = 0

    if len_b > len_a:
        m = (len_b - len_a) * ' '
        for i in a:
            m += str(i)
        for i, j in zip(*(b, m)):
            if i != j:
                cal += 1
        return cal

    elif len_b < len_a:
        m = (len_a - len_b) * ' '
        for i in b:
            m += str(i)
        for i, j in zip(*(a,m)):
            if i != j :
                cal += 1
        return cal
    else:
        for i,j, in zip(*(a,b)):
            if i != j:
                cal += 1
        return cal

def main():
    '''
    Do not change this code
    '''

    a = int(input())
    b = int(input())

    print(diffDigit(a, b))


if __name__ == "__main__":
    main()


