def zip_num(n):
    if len(str(n)) == 1:
        return n
    else:
        sum_num = 0
        for i in str(n):
            sum_num += int(i)
        if len(str(sum_num)) >= 2:
            return zip_num(sum_num)
        else:
            return sum_num
             
        
def main():
    n = int(input())
    print(zip_num(n))
    
if __name__ == "__main__":
    main()

