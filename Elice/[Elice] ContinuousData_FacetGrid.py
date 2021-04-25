from elice_utils import EliceUtils
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
elice_utils = EliceUtils()
train = pd.read_csv('data/train.csv')

def main():
    draw_facetgrid('Age')
    draw_facetgrid('Fare')
    

def draw_facetgrid(feature):
    # train에 저장된 DataFrame을 FacetGrid를 통해 그래프로 그려줍니다. 
    # hue="Survived"는 그래프의 범례(legend)의 이름을 설정합니다.
    # aspect=5 는 그래프의 종횡비를 설정해줍니다.
    facet = sns.FacetGrid(train, hue="Survived", aspect=5)
    
    # facet.map()은 kedplot 방식을 사용하여 주어진 데이터 feature를 plotting 하는 
    # 즉, 그래프를 그리는 기능을 합니다.  
    facet.map(sns.kdeplot, feature, shade=True)
    # 0 부터 값의 주어진 데이터의 최대 값까지를 x축의 범위로 설정합니다.
    facet.set(xlim=(0, train[feature].max()))
    # 지정된 범례(legend)를 표시.
    facet.add_legend() 
    plt.show()    
    
    # 에디터에서 출력을 위한 코드 수정 X
    plt.savefig("image.svg", format="svg")
    elice_utils.send_image("image.svg")

if __name__ == "__main__":
    main()

