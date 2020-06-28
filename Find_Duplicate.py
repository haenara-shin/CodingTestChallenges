def findDuplicate(nums):
    nums.sort()
    i = 0
    while True:
        if nums[i] == nums[i+1]:
            return nums[i]
        else:
            i += 1

def main():
    print(findDuplicate([1, 5, 2, 4, 5, 6, 3]))

if __name__ == "__main__":
    main()
