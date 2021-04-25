def correct(nums, solution):
    red = [i for i in range(1,6)]
    blue = [i for i in range(5, 0, -1)]
    yellow = [3, 3, 3, 3, 3]

    def color(color, nums, solution):
        key = []
        i = 0 
        while i < nums:
            if i >= len(red):
                key.append(color[i - len(red)])
            else: 
                key.append((color[i]))
            i += 1

        score = []
        for j in range(len(key)):
            if key[j] == solution[j]:
                score.append(True)
              
        return score.count(True)
    
    red_correct_count = color(red, nums, solution)
    blue_correct_count = color(blue, nums, solution)
    yellow_correct_count = color(yellow, nums, solution)
    

    max_val = max(red_correct_count,blue_correct_count,yellow_correct_count)
    print(max_val)
    
    result_dict = {'red':red_correct_count, 'blue':blue_correct_count, 'yellow':yellow_correct_count}
    
    final = []
    
    for k, v in result_dict.items():
        if v == max_val:
            final.append(k)
    
    if len(final) == 3:
        return final[0], final[1], final[2]
    elif len(final) == 2:
        return final[0], final[1]
    else:
        return final[0]
    

def main():
    nums = int(input())
    solution = list(map(int, input().split()))
    print(correct(nums, solution))
    
if __name__ == "__main__":
    main()
