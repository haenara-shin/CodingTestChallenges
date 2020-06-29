
def josephus(num, target):
    josephus = [i for i in range(1, num + 1)]
    result = []
    idx = target - 1


    for i in range(num):
        if len(josephus) > idx: # 위치 인덱스가 리스트를 넘지 않은 경우
            result.append(josephus.pop(idx)) 
            idx += target - 1 #다음 위치로 이동

        elif len(josephus) <= idx: # 위치 인덱스가 리스트를 넘은 경우
            idx = idx % len(josephus)
            result.append(josephus.pop(idx))
            idx += target - 1
    
    return result

def main():
    print(josephus(7, 3)) #[3, 6, 2, 7, 5, 1, 4]이 반환되어야 합니다

if __name__ == "__main__":
    main()
