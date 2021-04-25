from elice_utils import EliceUtils
import pandas as pd
import missingno as msno
import numpy as np
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import seaborn as sns
import pandas as pd
import warnings

elice_utils = EliceUtils()

def main():
    # data 를 불러옵시다. 데이터는 data 디렉토리안에 있습니다.
    train_data = pd.read_csv('data/train.csv')
    
    # missingno 라이브러리를 이용해서 null 데이터를 시각화해봅시다.
    # 실제 msno.matrix(train_data, figsize=(12,6)) 같은 역할 의 코드
    matrix(train_data, figsize=(12,12))



# missing 라이브러리의 그래프를 출력하기 위해 직접 missingno의 matrix 함수를 수정해서 사용.
def matrix(df,
           filter=None, n=0, p=0, sort=None,
           figsize=(25, 10), width_ratios=(15, 1), color=(0.25, 0.25, 0.25),
           fontsize=16, labels=None, sparkline=True, inline=False,
           freq=None):
    """
    missingno의 matrix 함수는 주어진 Datafrom의 Null 데이터를 매트릭스 형태로 시각화하는 동작을 합니다.
    최적의 성능을 위해 250행과 50열을 넘지 않도록 유의해주세요. 
    인자로 전달되는 각 파라미터들은 다음의 기능들을 수행합니다.
    df: mapping될 DataFrame 객체.
    filter: 히트 맵에 적용 할 필터입니다. "top", "bottom", None 중 하나의 값을 사용합니다. 기본값은 None입니다.
    n: 필터할 DataFrame에 포함 할 최대 열(columns) 수입니다.
    p: 필터할 DataFrame에서 열(columns)의 최대 채우기 비율입니다.
    sort: 히트 맵에 적용 할 정렬입니다. "오름차순(ascending)", "내림차순(descending)", None 중 하나의 값을 사용합니다. 기본값은 None입니다.
    figsize: 보여줄 할 이미지의 크기입니다.
    fontsize: 글꼴 크기. 기본값은 16입니다.
    labels: 열 이름을 표시할지 여부를 정하는 값입니다. 50개 이하의 열이있는 경우 기본 데이터 레이블로, 50개 이상인 경우에는 레이블이 없습니다.
    sparkline: 스파크 라인을 표시할지 여부를 결정하는 값입니다. 기본값은 True입니다.
    width_ratios: 매트릭스의 폭과 sparkline의 폭의 비율을 설정합니다. 기본값은 `(15, 1)` 입니다. 만약 `sparkline=False`일 경우에는 아무것도 하지 않습니다.
    color: 열(columns)의 색을 정합니다. 기본값은 `(0.25, 0.25, 0.25)` 입니다.
    반환값: 만약 `inline`이 False라면, `matplotlib.figure` 객체를 반환합니다. 그렇지 않다면, 아무것도 반환하지 않습니다.
    """
    df = nullity_filter(df, filter=filter, n=n, p=p)
    df = nullity_sort(df, sort=sort)

    height = df.shape[0]
    width = df.shape[1]

    # z는 컬러 마스크 배열, g는 NxNx3 매트릭스입니다. z색 마스크를 적용하여 각 픽셀의 RGB를 설정합니다.
    z = df.notnull().values
    g = np.zeros((height, width, 3))

    g[z < 0.5] = [1, 1, 1]
    g[z > 0.5] = color

    # matplotlib 그리드 레이아웃을 설정합니다.
    fig = plt.figure(figsize=figsize)
    if sparkline:
        gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios)
        gs.update(wspace=0.08)
        ax1 = plt.subplot(gs[1])
    else:
        gs = gridspec.GridSpec(1, 1)
    ax0 = plt.subplot(gs[0])

    # nullity plot을 생성합니다.
    ax0.imshow(g, interpolation='none')

    # 관계없는 시각적 요소들은 제거합니다.
    ax0.set_aspect('auto')
    ax0.grid(b=False)
    ax0.xaxis.tick_top()
    ax0.xaxis.set_ticks_position('none')
    ax0.yaxis.set_ticks_position('none')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)

    # column을 설정하고 회전합니다. labels은 기본적으로 없음으로 설정합니다. 
    # 그렇지 않다면 <= 50 열에 대해서는 표시하고 >50에 대해서는 표시하지 않습니다.
    if labels or (labels is None and len(df.columns) <= 50):
        ha = 'left'
        ax0.set_xticks(list(range(0, width)))
        ax0.set_xticklabels(list(df.columns), rotation=45, ha=ha, fontsize=fontsize)
    else:
        ax0.set_xticks([])
    
    #f req가 None이 아니면 Timestamps를 추가하고 그렇지 않으면 두 개의 맨 아래 행에 값을 설정합니다.
    if freq:
        ts_list = []

        if type(df.index) == pd.PeriodIndex:
            ts_array = pd.date_range(df.index.to_timestamp().date[0],
                                     df.index.to_timestamp().date[-1],
                                     freq=freq).values

            ts_ticks = pd.date_range(df.index.to_timestamp().date[0],
                                     df.index.to_timestamp().date[-1],
                                     freq=freq).map(lambda t:
                                                    t.strftime('%Y-%m-%d'))

        elif type(df.index) == pd.DatetimeIndex:
            ts_array = pd.date_range(df.index.date[0], df.index.date[-1],
                                     freq=freq).values

            ts_ticks = pd.date_range(df.index.date[0], df.index.date[-1],
                                     freq=freq).map(lambda t:
                                                    t.strftime('%Y-%m-%d'))
        else:
            raise KeyError('Dataframe index must be PeriodIndex or DatetimeIndex.')
        try:
            for value in ts_array:
                ts_list.append(df.index.get_loc(value))
        except KeyError:
            raise KeyError('Could not divide time index into desired frequency.')

        ax0.set_yticks(ts_list)
        ax0.set_yticklabels(ts_ticks, fontsize=20, rotation=0)
    else:
        ax0.set_yticks([0, df.shape[0] - 1])
        ax0.set_yticklabels([1, df.shape[0]], fontsize=20, rotation=0)

    # 컬럼 간격 간의 vertical grid를 생성합니다.
    in_between_point = [x + 0.5 for x in range(0, width - 1)]
    for in_between_point in in_between_point:
        ax0.axvline(in_between_point, linestyle='-', color='white')

    if sparkline:
        # 스파크 라인에 대한 row-wise completeness를 계산합니다.
        completeness_srs = df.notnull().astype(bool).sum(axis=1)
        x_domain = list(range(0, height))
        y_range = list(reversed(completeness_srs.values))
        min_completeness = min(y_range)
        max_completeness = max(y_range)
        min_completeness_index = y_range.index(min_completeness)
        max_completeness_index = y_range.index(max_completeness)

        # 스파크 라인을 설정하고 가장자리의 요소를 제거합니다.
        ax1.grid(b=False)
        ax1.set_aspect('auto')
        # GH 25
        if int(mpl.__version__[0]) <= 1:
            ax1.set_axis_bgcolor((1, 1, 1))
        else:
            ax1.set_facecolor((1, 1, 1))
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_ymargin(0)
        
        # plot sparkline---plot이 옆으로 있으므로 x 축과 y 축이 반전됩니다.
        ax1.plot(y_range, x_domain, color=color)

        if labels:
            # 아래의 코드는 label을 표시할 때 mixed, upper, lower중 어떠한 case를 사용하는지 찾아냅니다. 
            label = 'Data Completeness'
            if str(df.columns[0]).islower():
                label = label.lower()
            if str(df.columns[0]).isupper():
                label = label.upper()

            # 스파크 라인 lable을 설정하고 회전합니다.
            ha = 'left'
            ax1.set_xticks([min_completeness + (max_completeness - min_completeness) / 2])
            ax1.set_xticklabels([label], rotation=45, ha=ha, fontsize=fontsize)
            ax1.xaxis.tick_top()
            ax1.set_yticks([])
        else:
            ax1.set_xticks([])
            ax1.set_yticks([])

        # 최대 혹은 최소 labels를 추가합니다.
        ax1.annotate(max_completeness,
                     xy=(max_completeness, max_completeness_index),
                     xytext=(max_completeness + 2, max_completeness_index),
                     fontsize=14,
                     va='center',
                     ha='left')
        ax1.annotate(min_completeness,
                     xy=(min_completeness, min_completeness_index),
                     xytext=(min_completeness - 2, min_completeness_index),
                     fontsize=14,
                     va='center',
                     ha='right')

        ax1.set_xlim([min_completeness - 2, max_completeness + 2])  # Otherwise the circles are cut off.
        ax1.plot([min_completeness], [min_completeness_index], '.', color=color, markersize=10.0)
        ax1.plot([max_completeness], [max_completeness_index], '.', color=color, markersize=10.0)

        # tick mark를 제거합니다 (only works after plotting).
        ax1.xaxis.set_ticks_position('none')
        plt.savefig("image.svg", format="svg")
        elice_utils.send_image("image.svg")
    if inline:
        plt.show()
    else:
        return ax0
def nullity_sort(df, sort=None):
    """
    DataFrame의 퇴화차수(nullity)에 따라 오름차순 혹은 내림차순 순으로 정렬합니다.
    df : 정렬될 객체 DataFrame
    sort : 정렬할 방법, "ascending"(오름차순), "descending"(내림차순), or None(기본값)
    반환값 : 퇴화차수에 의해 정렬된 DataFrame
    """
    if sort == 'ascending':
        return df.iloc[np.argsort(df.count(axis='columns').values), :]
    elif sort == 'descending':
        return df.iloc[np.flipud(np.argsort(df.count(axis='columns').values)), :]
    else:
        return df


def nullity_filter(df, filter=None, p=0, n=0):
    """
    'top'과 'bottom'의 숫자 및 백분율 값의 조합을 사용하여 nullity에 따라 DataFrame을 필터링합니다. 백분율 및 숫자 임계 값(threshold)을 동시에 지정할 수 있습니다.
    예를 들어 75 % 이상의 완성도를 가진 DataFrame을 얻으려면 5 개 이하의 열을 사용하고, 
    nullity_filter (df, filter = 'top', p = .75, n = 5)를 사용할 수 있습니다.
    df: 정렬될 객체 DataFrame
    filter: DataFrame에 적용될 필터의 방향성. "상위"(top), "하위"(bottom), 혹은 None(기본값). 만약 해당 값을 비우거나 None이라 설정하면 filter는 DataFrame을 그대로 반환합니다.
    p: 완전성(completeness) 비율 한도. 만약 p가 0이 아니라면 필터는 최소 p의 완전성을 가진 열만을 찾아냅니다. 입력은 [0, 1] 범위에 있어야합니다.
    n: 수치 한도. 만약 n이 0이 아니라면 최대 n개의 열이 반환됩니다.
    반환값: nullity-filter가 적용된 `DataFrame`이 반환됩니다.
    """
    if filter == 'top':
        if p:
            df = df.iloc[:, [c >= p for c in df.count(axis='rows').values / len(df)]]
        if n:
            df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[-n:])]
    elif filter == 'bottom':
        if p:
            df = df.iloc[:, [c <= p for c in df.count(axis='rows').values / len(df)]]
        if n:
            df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[:n])]
    return df


if __name__ == "__main__":
    main()


