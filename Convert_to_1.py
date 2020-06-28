def convertTo1(num):
    d = [0]*(num+1)
    d[1] = 0
    for i in range(2, num+1):
        d[i] = d[i-1] + 1
        if i%2 == 0 and d[i] > d[i//2] + 1:
            d[i] = d[i//2] + 1
        if i%3 == 0 and d[i] > d[i//3] + 1:
            d[i] = d[i//3] + 1
    return d[num]

def main():
    print(convertTo1(10))

if __name__ == "__main__":
    main()
