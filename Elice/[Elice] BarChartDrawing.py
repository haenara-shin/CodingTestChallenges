from elice_utils import EliceUtils
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
elice_utils = EliceUtils()
train= pd.read_csv('data/train.csv')

# 코드를 지우고 직접 다시 작성해보세요!

def main():
    figure, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(nrows=2, ncols=3)
    figure.set_size_inches(18,12)
    bar_chart('Sex', ax1)
    bar_chart('Pclass', ax2)
    bar_chart('SibSp', ax3)
    bar_chart('Parch', ax4)
    bar_chart('Embarked', ax5)
    ax1.set(title="성별 생사정보")
    ax2.set(title="티켓 class")
    ax3.set(title="형제 수")
    ax4.set(title="부모 자식의 수")
    ax5.set(title="승선 장소")
    
    # 에디터에서 출력을 위한 코드 수정 X
    plt.savefig("image.svg", format="svg")
    elice_utils.send_image("image.svg")

def bar_chart(feature, ax):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, ax=ax)

if __name__ == "__main__":
    main()

