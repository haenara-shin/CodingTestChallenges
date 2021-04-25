def maxSubArray(nums):
    cache = [None] * len(nums)
    # 1.
    cache[0] = nums[0]

    # 2.
    for i in range(1, len(nums)):
        cache[i] = max(0, cache[i-1]) + nums[i]

    return max(cache)

def main():
    print(maxSubArray([-10, -7, 5, -7, 10, 5, -2, 17, -25, 1])) # 30이 리턴되어야 합니다

if __name__ == "__main__":
    main()
