import datetime
"""
날짜와 시간을 다루기 위한 클래스
"""
# 현재 시간과 날짜
today = datetime.datetime.now()
print(today)

# 출력값을 '요일, 월 일 연도'로 포매팅
print(today.strftime('%A, %B, %dth %Y'))

# 특정 시간과 날짜
pi_day = datetime.datetime(2020, 3, 14, 13, 6, 15)
print(pi_day)

# 두 datetime 의 차이
print(today - pi_day)

import os
"""
운영체제에 대한 정보 가져오기
"""
# 현재 어떤 계정으로 로그인 돼있는지 확인
print(os.getlogin())

# 현재 파일의 디렉토리 확인
print(os.getcwd())

# 현재 프로세스 ID 확인
print(os.getpid())

import os.path
"""
파일 경로 다룰때
"""

# 주어진 경로를 절대 경로로
print(os.path.abspath('..'))

# 주어진 경로를 현재 디렉토리를 기준으로 한 상대 경로로
print(os.path.relpath('현재 경로'))

# 주어진 경로들을 병합
print(os.path.join('경로1', '경로2'))

import re
"""
regular expression, 정규 표현식은 특정한 규칙/패턴을 가진 문자열을 표현
"""
pattern = re.compile('^[A-Za-z]+$')
print(pattern.match('I'))
print(pattern.match('love'))
print(pattern.match('python3'))

#숫자가 포함된 단어들만 매칭
pattern = re.compile('.*\d+')
print(pattern.match('I'))
print(pattern.match('love'))
print(pattern.match('python3'))

import pickle
"""
파이썬 오브젝트(객체)를 바이트 형식으로 바꿔서 파일에 저장할 수 있고,음
저장된 오브젝트를 읽어올 수 있
"""
# 딕셔너리 오브젝트
obj = {'my':'dictionary'}

# obj를 filename.pickle 파일에 저장
with open('filename.pickle', 'wb') as f:
    pickle.dump(obj, f)

# filename.pickle에 있는 오브젝트를 읽어옴
with open('filename.pickle', 'rb') as f:
    obj = pickle.load(f)

print(obj)

import json
"""
오브젝트를 JSON 형식으로 바꿔줌. 리스트-딕셔너리 만 바꿀 수 있음
"""

obj = {'my':'dictionary'}

# obj를 filename.json 파일에 저장
with open('filename.json', 'w') as f:
    json.dump(obj, f)

# filename.json 에 있는 오브젝트를 읽어옴
with open('filename.json', 'r') as f:
    obj = json.load(f)

print(obj)

import copy
"""
파이썬 오브젝트를 복사할때 쓰임.
"""
# = 연산자는 실제로 리스트를 복사하지 않음.
# 리스트를 복사하려면 슬라이싱을 사용하거나 copy.copy() 함수를 사용해야 함.
a = [1,2,3]
b = a
c = a[:]
d = copy.copy(a)
a[0] = 4
print(a, b, c, d)

# 하지만 오브젝트 안에 오브젝트가 있는 경우, copy.copy() 함수는 가장 바깥에 있는 오브젝트만 복사
# 오브젝트를 재귀적으로 복사하려면 copy.deepcopy() 함수를 사용해야 함

a = [[1,2,3], [4,5,6], [7,8,9]]
b = copy.copy(a)
c = copy.deepcopy(a)
a[0][0] = 4
print(a, b, c)

import sqlite3
"""
파이썬에서 SQLite 데이터 베이스를 사용할 수 있음
"""
conn = sqlite3.connect('example.db')

#SQL 문 실행
c = conn.cursor()
c.execute('''SELECT ... FROM ... WHERE''')
#가져온 데이터를 파이썬에서 사용
rows = c.fetchall()
for row in rows:
    print(row)
#연결종료
conn.close()