# 패키지를(모듈 파일 집어넣은 폴더) 임포트 하면 기본적으로 패키지 안에 있는 내용은 임포트 되지 않습니다.
# 패키지를 임포트할 때 패키지 안에 있는 애용도 함께 임포트 하고 싶다면 init 파일을 활용해야 함.

shape/__init__.py
"""
from shapes import area, volume

or 

from shapes.area import circle, square
"""

# __init__ 파일에서 변수를 정의 할 수도 있음. 상수 값을 init 안에 정의 하고 똑같이 from shapes import PI 이렇게 각 모듈에서 하면 됨.


# __all__ = ['모듈 이름', '상수 이름'] 이렇게 넣으면 import *로 불러올 수 있다. 안하면 안됨. 하지만 네임스페이스를 제대로 알아야 하니까 권장은 x.