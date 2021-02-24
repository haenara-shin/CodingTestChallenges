"""
Given an integer n. Each number from 1 to n is grouped according to the sum of its digits. 

Return how many groups have the largest size.

 

Example 1:

Input: n = 13
Output: 4
Explanation: There are 9 groups in total, they are grouped according sum of its digits of numbers from 1 to 13:
[1,10], [2,11], [3,12], [4,13], [5], [6], [7], [8], [9]. There are 4 groups with largest size.
Example 2:

Input: n = 2
Output: 2
Explanation: There are 2 groups [1], [2] of size 1.
Example 3:

Input: n = 15
Output: 6
Example 4:

Input: n = 24
Output: 5
 

Constraints:

1 <= n <= 10^4
"""
##

class Solution:
    def countLargestGroup(self, n: int) -> int:
      dict1 = {}
      for i in [str(i) for i in range(1, n + 1)]:
        temp = 0
        for j in i:
          temp += int(j)
        if temp in dict1:
          dict1[temp] = dict1[temp] + 1
        else:
          dict1[temp] = 1
      result = list(dict1.values())
      max_idx = max(result)
      return result.count(max_idx)


class Solution:
    def countLargestGroup(self, n: int) -> int:
        result = []
        c = [str(i) for i in range(1, n + 1)]
        for i in c:
            temp = 0
            for j in i:
                temp += int(j)
            result.append(temp)
        from collections import Counter
        r = list(Counter(result).values())

        return r.count(max(r))

class Solution:
    def countLargestGroup(self, n: int) -> int:
        result = []
        for i in range(1,n+1):
            x = sum(list(map(int, list(str(i)))))
            result.append(x)
        from collections import Counter
        a = list(Counter(result).values())
        return a.count(max(a))
    
    



