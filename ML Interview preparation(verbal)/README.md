(Updated on 25 Aug. 2021)

# Table of Content  -- the link will be fixed soon.
1. [ML Interview preparation (verbal)](#ML-Interview-preparation-(verbal))
2. [대학원 진학의 경우](#대학원-진학의-경우)
3. [퀵 정리](#퀵-정리)
4. [Linear Algebra](#Linear-Algebra)
   1. Linearly Independent (선형 독립)?
   2. Basis (기저), Dimension (차원), Rank(계수), Determinant (행렬식)?
   3. Null space 란 무엇?](#Null-space-란-무엇?
   4. Symmetric matrix (대칭 행렬)?
   5. Positive-definite 란?
   6. Eigenvector(고유벡터), Eigenvalue(고유값), Diagonalization(대각화), SVD(특이값 분해)?
   7. Jacobian matrix 란 무엇?
5. [Statistics/Probability (확률/통계) 및 Statistical Learning](#Statistics/Probability-(확률/통계)-및-Statistical-Learning)
   1. 확률이랑 통계 다른 점
   2. Central Limit Theorem 이란? Central Limit Theorem 은 어디에 쓸 수 있는지?
   3. 큰 수의 법칙 이란?
   4. (Joint, Marginal, Conditional) Probability Distribution 이란 무엇인가
   5. Bayes rule/theory 란 무엇인가?
   6. Bias 란 무엇인가? Unbiased estimation은 무조건 좋은가? Unbiased estimation의 장점은 무엇?
   7. Biased/Unbiased estimation의 차이는?
   8. Bias, Variance, MSE란? 그리고 그들의 관계는?
   9. Sample Variance (표본 분산) 란 무엇인가?
   10. Sample variance(표본 분산)를 구할 때, N 대신에 N-1로 나눠주는 이유는?
   11. Gaussian Distribution에서 MLE와 Sample Variance 중에 어떤 걸 사용해야 하는가? (정확한 정답인지는 모르겠음. 질문이 잘 이해가 안됨.)
   12. [Binomial (이항), Bernoulli(베르누이), Multinomial(다항), Multinoulli 란 무엇인가?]
   13. [Beta, Dirichlet, Gamma, Poisson Distribution은 어디에 쓰이는가?]
   14. [Conjugate Prior Distribution (켤레 사전 분포) 란?]
   15. [Bias and Variance Trade-off 란 무엇인가?]
   16. [Confidence interval (신뢰구간) 이란?]
   17. [Covariance/correlation 이란?]
   18. [Total variation (SST), Explained variation (SSR), Unexplained variation(SSE) 이란?]
   19. Coefficient of determination 이란? (결정 계수, $R^2$)
   20. P-value 란?
   21. Likelihood-ratio test (우도비 검정법)?
   22. KDE(Kernel Density Estimation)?
   23. 모수 추정(Parameter estimation)
6.  [Machine Learning](#Machine-Learning)
    1.  [Frequentism 와 Bayesian 차이/장점]
    2.  [차원의 저주란?]
    3.  [Train, Valid, Test를 나누는 이유는 무엇? 교차 검증(cross validation) 이란?]
    4.  [(Super-, Unsuper-, Semi-super)vised learning 이란?]
    5.  [Receiver Operating Characteristic Curve 란?]
    6.  [Precision, Recall, Type I/II error]
    7.  [Precision Recall Curve?]
    8.  [Information?]
    9.  [Entropy 란?]
    10. [Cross Entropy 와 KL-Divergence 란?]
    11. [Mutual Information 이란?]
    12. [Cross-Entropy loss 란?]
    13. [Decision theory란?]
    14. [Generative model vs. Discriminative model 이란?]
    15. [분류와 회귀의 차이]
    16. [회귀란? (Regression)]
    17. [Overfitting/Underfitting 이란 무엇이고 어떤 문제?]
    18. [Overfitting 과 Underfitting 을 해결하는 방법은?]
    19. [Regularization 이란?]
    20. [Lasso/Ridge/ElasticNet 이란?]
    21. [Regularization_version2]
    22. [Activation function 이란? 3가지 activation function type]
    23. [CNN 에 대해 설명]
    24. [RNN 에 대해 설명]
    25. [Newton's method 이란 무엇?]
    26. [경사하강법(Gradient Descent)]
    27. [역전파(Backpropagation)]
    28. [Optimizer 의 종류와 차이]
    29. [Local optimum 으로 빠지는데 성능이 좋은 이유는 무엇?]
    30. [Internal Covariate Shift]
    31. [Batch Normalization 은 무엇이고 왜 하는지?]
    32. [BatchNormalization vs. LayerNormalization]
    33. [Ensemble 이란?]
    34. [Stacking 이란?]
    35. [Bagging 이란?]
    36. [Boosting 이란?]
    37. [Bagging 과 Boosting 차이?]
    37. [AdaBoost/Logit Boost/Gradient Boost/LGBM 차이]
    38. [SVM 이란?]
    39. [Margin 을 최대화 하면 어떤 장점?]
    40. [PCA 란?]
    41. [LDA 란? (Linear Discriminant Analysis)]
    42. [SVD (Singular Value Decomposition) 란? (선형 대수학 섹션에서 다시 설명)]
    43. [Clustering 종류]
    44. [U-net 구조 및 특징]
    45. [트리 기반 모델의 Feature importance 어떻게 구함?])
    46. [Cut-mix 가 잘되는 이유?]
7.  [수행 했던 프로젝트들 중 기술 스택](#수행-했던-프로젝트들-중-기술-스택)
    1.  [EfficientNet의 우월함?]
    2.  [R-CNN vs. Fast R-CNN vs. Faster R-CNN]
    3.  [mAP in Image detection]

#
## ML Interview preparation (verbal)
- 혼자서 정리해보는 ML 인터뷰 준비
- 질문은 [여기서](https://jrc-park.tistory.com/m/259) 주로 발췌했으며, 답안은 내가 아는 대로 붙임.
#

## 대학원 진학의 경우
1. 자기소개/지원동기/연구계획 요약 발표
2. 왜 해당 대학원을 선택했는지?
3. 연구 계획 및 졸업 후 진로
4. 진행했던 프로젝트 관련 질문 및 해당 분야에 대해 얼마나 알고 있는지
5. 학사과정 이수표에서 잘 나온 과목 중 관련 분야 교수님이 전공 질문할 수도 있음
6. 졸업 후 박사 진학할 생각이 있는지
7. 원하는 연구실 들어가지 못한다면 진학하지 않을것인지
8. 하고 싶은 말


#
## 퀵 정리
- [x] Parameter?
  - y = mx + c 에서 m과 c. 즉 `가중치(weight)` (부연설명 필요 없을듯 ㅋ)
- [x] MLE?
  - 모델의 파라미터 값을 결정하는 방법 중 하나.
  - 파라미터 값으로 만들어진 모델에서 만들어진 예측 값이 실제 관측된 데이터를 생산하는 `가능성`을 최대화 하는 것을 찾음.
- [x] log 사용 이유
  - log는 단조증가함수이므로 log를 취했을 때 최대값을 가지는 지점과 원래 최대값을 가지는 지점이 동일하고, 보통 곱셈보다 덧셈이 계산이 더 간편하며, 미분이 쉽다.


#
## Linear Algebra
#
- [x] Linearly Independent (선형 독립)?
  
  > 같은 수의 성분을 가진 n개의 벡터 a<sub>1</sub>, a<sub>2</sub>, ..., a<sub>n</sub>에 대해서 이들 벡터의 1차 결합(linear combination)인 c<sub>1</sub>a<sub>1</sub> + c<sub>2</sub>a<sub>2</sub> + ... + c<sub>n</sub>a<sub>n</sub> = 0 을 만족하는 상수 c<sub>1</sub>, c<sub>2</sub>,... , c<sub>n</sub> 이 모두 0 이면, 벡터 a<sub>1</sub>, a<sub>2</sub>, ..., a<sub>n</sub> 은 linear independent 임.
- [x] Basis (기저), Dimension (차원), Rank(계수), Determinant (행렬식)?
  - `Basis`
    - 어떤 vector space V(그냥 벡터의 집합임..)의 벡터들이 linearly independent 하면서 vector space V 전체를 생성할 수 있는 벡터들의 집합.
      - vector space가 되기 위해선 +, *에 닫혀있고, 이것들의 선형 결합에도 닫혀 있어야 함.
    - $R^m$의 임의의 원소를 표현하기 위해 필요한 최소한의 벡터로 이루어진 집합
    - m 차원의 임의의 원소를 표현하는데 필요한 최소한의 벡터가 m개. (m개 초과하면 basis는 아님)
    - 즉, 선형 독립은 제로 벡터([0,0,...,0])에 한정된 개념이고, 기저는 m 차원의 모든 벡터를 대상으로 하는 개념
    > [y<sub>1</sub>,y<sub>2</sub>,...,y<sub>m</sub>] = c<sub>1</sub>[a<sub>11</sub>,a<sub>21</sub>,...,a<sub>m1</sub>] + c<sub>2</sub>[a<sub>12</sub>,a<sub>22</sub>,...,a<sub>m2</sub>] + ... + c<sub>n</sub>[a<sub>1n</sub>,a<sub>2n</sub>,...,a<sub>mn</sub>] 의 해가 1개만 존재하는 경우, 집합 [[a<sub>11</sub>,a<sub>21</sub>,...,a<sub>m1</sub>], [a<sub>12</sub>,a<sub>22</sub>,...,a<sub>m2</sub>], ..., [a<sub>1n</sub>,a<sub>2n</sub>,...,a<sub>mn</sub>]]을 $R^m$의 Basis 라고 함.
  - `Dimension`
    - vector space V에 속한 linearly independent 한 vector들의 최대수를 V의 차원(dimension, `dim V`)이라고 함.
    - W가 $R^m$의 부분공간(subspace)이고, 벡터 a_1, a_2,...,a_n이 W의 linearly independent 원소 라고 할 때, basis의 원소의 개수를 subspace W의 dimension (`dim n`)이라고함.
      - 모든 subspace는 반드시 원점(origin)을 지나야 함.
  - `Rank`
    - row echelon form 만들었을때 0으로 채워진 행 제외한 행의 갯수
    - $R^m$의 n개의 벡터 [[a<sub>11</sub>,a<sub>21</sub>,...,a<sub>m1</sub>], [a<sub>12</sub>,a<sub>22</sub>,...,a<sub>m2</sub>], ..., [a<sub>1n</sub>,a<sub>2n</sub>,...,a<sub>mn</sub>]] 중 linearly independent 한 벡터의 갯수
  - `Determinant`
    - 역행렬이 존재하는지 여부를 확인하는 방법
    - `det(A) != 0 --> 역행렬 존재 --> non-singular matrix` (n차 정방행렬) // 역행렬이 존재하지 않는 행렬은 특이 행렬(singular matrix)
    - `선형 변환 할때 단위 면적이 얼마만큼 늘어나는가`
- [x] Null space 란 무엇?
   - 선형 방정식 Ax=b 에서 b가 zero vector(null vector) 일때 식을 만족시키는 모든 가능한 해 x 에 대한 집합. 한마디로, `Ax=0 의 해(solutions)들이 이루는 공간`
   - 어떤 null space 든지 반드시 zero vector는 포함.
     - `Column space`는 Ax=b 에 대해 해가 존재 하는 x의 집합. b 벡터가 A의 column의 선형 결합으로 표현이 가능함.
- [x] Symmetric matrix (대칭 행렬)?
  - $A^T = A$
    - 조건 1. a<sub>ij</sub> = a<sub>ji</sub>, 
    - 조건 2. 정방행렬
    - 특징 1. 대칭 행렬의 고유값(eigenvalue)는 전부 실수(real number)를 가지며 각각의 고유값에 대응되는 고유 벡터들은 1차원의 고유공간을 형성
    - 특징 2. 대칭 행렬의 고유벡터(eigenvector)는 직각(perpendicular)을 이룸.
- [x] Positive-definite 란?
    - 성분이 모두 실수(real number), 대칭 행렬(정방 행렬 A = $A^T$)이 x != 0 인 모든 벡터에 대해서 $x^TAx>0, A>0$ 만족
      - positive-semidefinite 는 x = 0 인 조건도 포함
    - 모든 고유값(eigenvalue)이 0 보다 큼.
- [x] Eigenvector(고유벡터), Eigenvalue(고유값), Diagonalization(대각화), SVD(특이값 분해)?
  - `정방행렬 A`에 대해서 $Ax=\lambda x$ 가 성립하는 `0이 아닌 벡터 x를 고유벡터(eigenvector)` 그리고 `상수`($\lambda$)를 `고유값(eigenvalue)` 라고 함.
  - 기하학적 의미: 행렬의 곱의 결과가 `원래 벡터와 '방향'은 같고`, `'배율'만 상수`($\lambda$) `만큼 비례해서 변함`.
  - `Diagonalization(대각화) eigenvalue decomposition`: Eigenvalue decomposition 은 정방행렬만 적용 가능하며, D 행렬에 고유값이 들어가 있음.
    - $A = PDP^-1 = PDP^T$, P = eigenvectors, D = eigenvalues(diagonal matrix)
  - SVD(Singular Value Decomposition) 특이값 분해
    - 정방 행렬 뿐만 아니라 m x n 직사각 행렬(rectangluar matrix)에 폭넓게 사용 가능.
    - $A = U\sum V^T$, 
      - A = m x n rectangular matrix, 
      - U = A의 left singular vector로 이루어진 m x m orthogonal matrix, 
      - $\sum$ = 대각 성분이 $\sqrt{\lambda_i}$ 로 이루어진 m x n diagonal matrix,
      - V = A의 Right singular vector로 이루어진 n x n orthogonal matrix
      - U = $AA^T$, V = $A^TA$ 의 고유벡터와 고유값 구함
    - 고유값 분해를 통한 `대각화의 경우, 고유벡터의 방향은 변화가 없고 크기만 고유값 만큼 변함`. 하지만, `특이값 분해는 방향과 크기 모두 변함` 방향은 (U, $V^T$)에 의해 변하고, 크기 변화는 $\sum$(특이값)들 만큼 변함.
      - SVD는 차원 축소를 하는데 사용.
        - 1. 데이터를 묶어서 A
        - 2. 행은 몇 번째 데이터인지, 열은 feature 나타냄
        - 3. 각 열의 평균을 0으로 만듦.
        - 4. V의 열 벡터는 데이터들을 사영시켰을때 분산이 가장 커지는 축들을 열로 나타냄 (가장 왼쪽, 즉 첫 번째 열이 데이터들의 분산이 제일 커지도록 하는 축. 왼쪽 열에 가까운 열일 수록 데이터의 분산이 잘 된 축.)
        - 5. AV (= U$\sum$) 의 행 벡터는 새로운 축에 대한 좌표 표현. 열 벡터는 데이터 1~n을 축 열(축) 1, 2, 3...을 기준으로 표현한 좌표값들의 모임.
        - 6. AV에서 적당한 크기로 열을 잘라내면 중요한 정보를 가급적 유지한 채(데이터들의 분산이 커지도록 축을 설정) 차원이 줄어듦. 
      - [Ref](https://luminitworld.tistory.com/69)
- [x] Jacobian matrix 란 무엇?
  - [Ref](https://angeloyeo.github.io/2020/07/24/Jacobian.html)
  - 행렬의 모든 원소들은 1차 미분 계수로 구성
  - (국소/미소 영역에서) `비선형 변환을 선형 변환으로 근사 시킨 것`
    - 선형 변환?
      1. 변환 후에도 원점의 위치가 변하지 않고,
      2. 변환 후에도 격자들의 형태가 직선의 형태를 유지하고 있으며,
      3. 격자 간의 간격이 균등해야함.
     - 행렬식(`determinant`)는 선형 변환 할 때 `단위 면적이 얼마만큼 늘어나는가` 의미함. `Jacobian의 행렬식 의미`도 `원래 좌표계에서 변환된 좌표계로 변환될 때의 넓이의 변화 비율`
#
## Statistics/Probability (확률/통계) 및 Statistical Learning
#
- [x] 확률이랑 통계 다른 점
  - 확률: 모집단에 관한 데이터 제공
  - 통계: 특정 이벤트를 기반으로 하며 표본 속성은 모집단의 특성을 추론하는데 사용. 관찰/데이터를 기반.
- [x] Central Limit Theorem 이란? Central Limit Theorem 은 어디에 쓸 수 있는지?
  - 모집단이 평균이 $\mu$ 이고 표준편차가 $\sigma$ 인 임의의 분포 일때, 이 모집단으로부터 추출된 표본의 `표본의 크기 n이 충분히 크다면` 표본 평균들이 이루는 분포는 평균이 $\mu$ 이고 표준편차가 $\sigma/ \sqrt{n}$ 인 정규분포에 근접함.
  - 모집단이 어떤 분포를 가지고 있든지(모집단 분포가 어떤 모양이든 상관 없이 - 균등 분포, 비균등 분포, or 정규 분포) `일단 표본의 크기가 충분히 크다면, 표본 평균들의 분포가 모집단의 모수를 기반으로한 정규분포를 이룬다`. 이런 점을 이용해서 `특정 사건`(내가 수집한 표본의 평균)`이 일어날 확률 값을 계산`할 수 있음. 
    - 좀 더 어려운 말로...표본 평균들이 이루는 표본 분포와 모집단 간의 관계를 증명함. 수집한 표본의 통계량(statistics)을 이용해 모집단의 모수(parameters)를 추정할 수 있는 확률적 근거 마련.
    - 좀 쉬운 말로...`"모집단 분포에 상관 없이" 큰 표본들의(표본의 크기는 최소 30 이상)` `"표본 평균의 분포"`가 `"정규분포로 수렴"`
  - 가능 1. 샘플 수가 30개 이상이면 모집단의 평균과 분산을 알아낼 수 있다.
  - 가능 2. 모든 샘플들은 정규분포로 나타낼 수 있으며, 정규분포와 관련된 수학적 이론들을 적용할 수 있음.
- [x] 큰 수의 법칙 이란?
  - 사건을 무한히 반복할 때 일정한 사건이 일어나는 비율은 횟수를 거듭하면 할수록 일정한 값에 가까워지는 법칙 (샘플링은 복원 추출임)
  - 어떤 모집단에서 표본들을 추출할 때, 각 표본의 크기가 커지면(시행 횟수가 늘어나면) 상대도수와 모비율의 값이 같아질 확률이 높아짐.
- [x] (Joint, Marginal, Conditional) Probability Distribution 이란 무엇인가
  - 1. `PMF` vs. `PDF`
    - PMF: `이산확률(동전 앞뒤, 주사위 눈 등 countable, 셀 수 있는)변수의 확률분포`를 나타내는 함수 ($\sum$)
    - PDF: `연속확률(돈, temperature, time 등 셀 수 없는)변수의 확률분포`를 나타내는 함수 ($\int$)
  - 2. `Joint Probability Distribution` (결합 확률 분포)
    - 두 개 이상의 사건이 동시에 일어날 확률에 대한 분포
    - $P(x,y) = P(X \cap Y)$ (= P(x) x P(y) if 두 사건이 독립 사건이라면)
  - 3. `Marginal Probability Distribution` (주변 확률 분포)
    - 두 확률 변수 x, y에 대한 Joint Probability Distribution $P_{x,y}(x,y)$ 에서 x에 대한 확률 분포를 구하고자 한다면, y를 중요하지 않은 변수로 취급(marginalize) 함. (두 개의 변수로 이루어진 결합확률분포를 하나의 변수로 표현하기 위함)
  - 4. `Conditional Probability Distribution` (조건부 확률 분포)
    - 두 개의 확률 변수 (또는 사건) x, y 에 대하여 결합확률분포와 주변확률분포를 이용해 구함.
      - 조건부확률분포 = (결합확률분포) / (주변확률분포)
    - (PMF 일때의 예시) $P_{x|y}(x_i|y_j) = \frac {P_{x,y}(x_i, y_j)}{P_y(y_j)}$ 
- [x] Bayes rule/theory 란 무엇인가?
  - 사후확률 = 가능도 x 사전확률 / Marginal(상수)
  - 새로운 정보를 토대로 어떤 사건이 발생했다는 `주장에 대한 신뢰도`(베이지안의 `확률`)를 갱신해 나가는 방법
#
- [x] Bias 란 무엇인가? Unbiased estimation은 무조건 좋은가? Unbiased estimation의 장점은 무엇?
   > 예측 값(추정된 파라미터들의 `평균`)과 실제 값(실제 파라미터)의 차이. 
   > > 상식적으로 '예측 값 - 실제 값' 이라고 생각하자.
- [x] Biased/Unbiased estimation의 차이는?
   - bias 는 `평균` 값의 차이를 다루기 때문에 같은 bias를 가진다고 하더라도 추정된 값들 끼리 차이가 클수도/작을수도 있다.
     - Unbiased estimation: 파라미터 추정 평균에 대해서 bias가 0 인 경우.
       - 모수(parameter)를 중심으로 하는 표본 분포(sampling distribution)을 가짐. `분산을 바탕으로한 유효성 비교시 쓸모 있음(어느 추정량의 분포가 더 모수에 집중적으로 분포하는지)`.
       - 하지만 이러한 점추정(point estimate)은 표본으로부터 구한 단일의 값이기 때문에, 점추정 값은 `그 추정값이 모수에 얼마나 가까운지에 대한 정보를 포함하지 않음`. 점추정은 관측된 표본들로부터 추론된 하나의 값으로만 구해지기 때문에, `표본의 크기가 작을 경우 모집단의 특성을 잘 반영하지 못하는 경우가 생길 수 있음`. --> 그래서 구간 추정을 사용.. 
        - (ex) Gaussian distribution 에서 반복적으로 x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub> 값들을 뽑아서, 첫 번째 원소로 Gaussian의 평균을 추정하는 것 
      - Biased estimation: 파라미터 추정 평균의 bias가 0이 아닌 경우    
- [x] Bias, Variance, MSE란? 그리고 그들의 관계는?
  - `Bias`: $bias(\hat y) = E_\theta (\hat y) - y$ ...(`예측한 값의 평균과 실제값의 차이`)(`오차`)
  - `Variance`: $variance(\hat y) = E[(\hat y - E(\hat y))^2]$ ...(`예측값이 예측값의 평균을 중심으로 얼마나 퍼져있는지`)
  - `MSE`: $E_y (\hat y - y)^2$ ... (예측값 - 실제값)의 제곱에 대한 기대값 (오차를 제곱한 값의 평균...정확도)
- [x] Sample Variance (표본 분산) 란 무엇인가?
  - 모집단에서 무작위로 n개의 표본을 추출했을 때, n개 표본들의 평균/분산을 각각 'sample mean', 'sample variance' 라고 함.
- [x] Sample variance(표본 분산)를 구할 때, N 대신에 N-1로 나눠주는 이유는?
  - 참고로, 표본 평균은 상식적으로 확률 표본의 합을 n으로 나눔. 하지만 표본 분산은 왜 n-1로 나눌까?
  - 기본 조건 4가지
    1. n개의 표본을 추출할 때, 각 표본은 서로 독립
    2. 각 표본은 같은 확률 분포를 가짐
    3. E(x_1) = E(x_2) = ... = E(x_n) = E(x) = $\mu$ , x_1, x_2..., x_n을 n개의 확률 표본이라고 함
    4. V(x_1) = V(x_2) = ... = V(x_n) = V(x) = $\sigma^2$
  - 표본은 모집단의 부분집합 이기 때문에 표본 분산의 기대값은 모집단의 분산으로 표현하기 위해서는 나누는 값이 n이 아니라 n-1을 사용함. 만약 n-1이 아니라 n으로 나눴으면 표본 분산의 기대값은 모분산이 나오지 않음. (즉, `표본 분산의 기대값 = 모분산 이기 때문`)
    - $E(S^2) = \sigma^2$
- [x] Gaussian Distribution에서 MLE와 Sample Variance 중에 어떤 걸 사용해야 하는가? (정확한 정답인지는 모르겠음. 질문이 잘 이해가 안됨.)
  - 가우시간 분포로 추정되었을 때 최대가능성은 데이터 포인트들이 평균값에 더 가까울때 발견되는데, 이는 가우시안 분포가 좌우대칭이기 때문. 따라서 데이터 포인트와 평균 값 간의 거리를 최소화 하는 것과 동일하다. 

#
- [x] Binomial (이항), Bernoulli(베르누이), Multinomial(다항), Multinoulli 란 무엇인가?
    1. `Binomial(이항)` 분포
        - 베르누이 시행을 n번 반복
        - n회 베르누이 시행 중 x번 성공할 확률
    2.  `Bernoulli(베르누이)` 분포
        - 성공(p) 또는 실패(1-p) 두 가지 중 하나로만 나오는 경우. 1회 성공 확률이 p인 독립 시행을 반복해서 시행.
    3.  `Multinomial(다항)` 분포
        - 여러 개의 값을 가질 수 있는 독립 확률 변수들에 대한 분포 (여러 번의 독립 시행에서 각각의 값이 특정 횟수가 나타날 확률)
    4.  `Multinoulli(categorical distribution 이라고도 함)` 분포
        - 1 부터 k 까지의 k개의 정수 값 중 하나가 나오는 분포
- [x] Beta, Dirichlet, Gamma, Poisson Distribution은 어디에 쓰이는가?
    1. `Beta` 분포
        - 0 과 1 사이의 값을 가지는 단일(univariate) 확률 변수의 베이지안 모형에 사용
    2. `Dirichlet` 분포
        - 0 과 1 사이의 값을 가지는 다변수(multivariate) 확률 변수의 베이지안 모형에 사용. (다변수 확률 변수들의 합이 1이 되어야함.) 
    3. `Gamma` 분포
        - 대기 시간이 얼마나 되는지, 어떤 사건이 발생할 때 까지 얼마나 많은 시간이 필요한지 등에 사용되어 신뢰도에 적용
        - 어떤 사건이 일정 간격 동안 발생 횟수의 평균이 1/$\beta$ 로 주어질때, $\alpha$번 발생했을 시간(대기 시간)에 대한 확률 분포
    4. `Poisson` 분포
        - 정해진 시간 안에 어떤 사건이 일어날 횟수에 대한 기대값($\lambda$)을 알고 있을때, 그 사건이 n회 일어날 확률
        - 어떤 사건이 일정 간격 동안 평균 $\mu = \lambda t$ 발생 했을때, x번 발생할 확률
- [x] Conjugate Prior Distribution (켤레 사전 분포) 란?
  - [Ref](https://rooney-song.tistory.com/11)
  - 사전분포(Prior distribution)와 사후분포(Posterior distribution)가 동일한 분포족(보통 exponential family)에 속하면 사전분포를 켤레사전분포 라고 함.
  - 켤레사전분포를 사용하는 이유는 사후분포의 계산이 편리해지기 때문.
  - ![스크린샷 2021-08-15 오후 5 31 59](https://user-images.githubusercontent.com/58493928/129497539-d9cbdee6-d668-4291-bb64-923282fa5249.png)
- [x] Bias and Variance Trade-off 란 무엇인가?
  > 추정된 파라미터 값들의 차이가 클수록 값들이 멀리 퍼져있지만(큰 variance) 평균은 실제 파라미터와 비슷해짐(작은 bias)
  >> $$ MSE = E[({\hat \theta - \theta})^2 ]  = Bias + Variance $$
    >>> $$ Bias((E[\hat \theta] - \theta)^2) $$
    >>> $$ Variance = E[(E[\hat \theta] - \hat \theta)^2]$$
    - MSE가 동일하다면, Bias 값에 따라서 Variance 도 달라지게 되고 그 반대도 성립함.
    - 머신러닝에서 학습을 계속하면, 추정되는 파라미터는 실제 파라미터와 차이가 줄어들게 됨. 하지만 파라미터에 대해서 Bias 가 낮은것 보다는 MSE가 작은 경우가 제일 좋은 것임. 
    - Variance 를 높이려면 모델의 complexity를 높이면 됨. 
  #
- [x] Confidence interval (신뢰구간) 이란?
  - 모비율이 해당 구간 사이에 존재할 확률이 confidence interval 임.
  - ex) 95% 신뢰구간(신뢰수준)으로 A 후보 지지율의 모비율이 37%-43% 사이에 존재함.
- [x] Covariance/correlation 이란?
  - `공분산(covariance)`: 확률변수가 2가지 이상일때, 각 확률 변수들이 어떻게 퍼져 있는지 나타냄. X가 커질때 Y는 어떻게 되는지? 
    - 공분산이 0 이라면 두 변수간에는 아무런 선형 관계가 없으며 서로 독립적인 관계에 있음. 그러나, 항상 독립적인 관계는 아님.
    - 독립적이면 항상 공분산이 0
    - `X의 편차와 Y의 편차를 곱한 것의 평균(기대값)`
      - $Cov(x,y) = E((x-\mu)(y-\nu)) = E(xy) - \mu \nu$
        - $E(x) = \mu, E(y) = \nu$
  - `상관계수(Correlation)`: 공분산은 x와 y 단위의 크기에 영향을 받음. 따라서 `공분산을 분산의 크기만큼 나눔`.
    - $R = \frac {Cov(x,y)}{\sqrt{Var(x)Var(y)}}$
    - 확률 변수 x, y가 독립이면 상관계수는 0
- [x] Total variation (SST), Explained variation (SSR), Unexplained variation(SSE) 이란?
  - `SST` = `SSE` + `SSR`
  - `Total` = `Explained (SSR)` + `Unexplained (SSE)`
  - Sum of Squares `Total` = Sum of Squares `Regression` + Sum of Squares `Error`
  - $\sum(y-\bar y)^2 = \sum (\hat y - \bar y)^2 + \sum(y - \hat y)^2$
    - $\bar y$ = Average value of the dependent variable (평균값)
    - $\hat y$ = Estimated value of y for the given x value (예측값)
    - y = Observed values of the dependent variable (실제값)
- [x] Coefficient of determination 이란? (결정 계수, $R^2$)
  - `선형 회귀 분석에서 회귀 직선의 적합도(goodness-of-fit)를 평가`하거나 종속변수에 대한 설명 변수들의 설명력을 알고자 할 때.
  - $R^2 = \sum(\hat y_i - \bar y)/{\sum(y_i - \bar y)^2}$ = 회귀선에 의해 설명되는 변동 / 전체 변동 = 상관계수$^2$
- [ ] Total variation distance 란?
- [x] P-value 란?
  - 검정 통계량에 관한 확률. 우리가 얻은 검정 통계량 보다 크거나 같은 값을 얻을 수 있을 확률.
    - 검정 통계량들은 거의 대부분 `귀무가설`을 가정하고 얻게 되는 값임. 즉, 두 표본 집단의 모집단은 같다는 가정을 전제함.
      - 귀무가설(null hypothesis): `실패를 기도하는 이론`. 새로울것이 없다, 는 가정. 무죄추정의 원칙 같음. "흡연 여부가 뇌 혈관 질환의 발생에 영향을 미치지 않는다" 라고 가정. --> 대립가설: 영향을 미친다.
      - 귀무가설로는 결과 해석불가 --> 귀무가설 기각 --> 대립가설 채택 
    - 우리가 얻은 데이터에 있는 두 표본 집단이 같은 모집단에서부터 나온거라면, 검정 통계량(가령 t-value)을 얻었는데 이게 얼마나 말이 되는건지? 보통 5% 기준을 많이 사용하며, p-value 가 5% 보다 작으면 유의한 차이가 있다고 말함.
      - 하지만 표본의 크기가 커지거나 효과의 크기(effect size, ex>표본 평균간의 차이) 커지거나 둘 중 하나만 변하더라도 p-value가 마치 유의미한 차이를 담보할 수 있을것 마냥 작아지기 때문에 맹신할 수 없다.
- [x] Likelihood-ratio test (우도비 검정법)?
  - 모형 두 개의 우도 비를 계산해서 두 모형의 우도가 유의하게 차이나는지 비교하는 방법
  - 회귀 모형에 변수를 추가 또는 제거하면서 기존과 새로운 모형에서의 우도 비를 통해 회귀 계수의 유의성을 검정하면서 변수의 유효성을 판단함.
  - ![스크린샷 2021-08-15 오후 9 38 15](https://user-images.githubusercontent.com/58493928/129512100-c49ef3c6-6f51-46a3-86bf-1e85f1badb08.png)
- [x] KDE(Kernel Density Estimation)?
  - 커널 함수를 이용한 밀도 추정 방법의 하나. [Ref](https://darkpgmr.tistory.com/147?category=761008)
    - `밀도 추정(Density Estimation)`이란?
      - 여기서의 `밀도`는 확률 밀도(probability density)의 함수 값을(일정 구간에 대한 적분 값) 의미함. 즉, 밀도를 추정한다고 하면 확률 밀도 함수를 추정하는 것.  즉, 어떤 변수의 `분포의 특성을 추정`.
        - 밀도 추정 방법은 `parametric` vs `non-parametric` 2가지 방법이 있음.
          1. `Parametric` 추정: `미리 PDF 모델을 정해놓고` 데이터로부터 모델의 파라미터만 정함. 하지만 현실에서는 모델을 미리 아는 경우가 거의 없음...ex> '일일 교통량'이 정규분포를 따른다고 가정해 버리면 관측된 데이터들로부터 평균과 분산만 구하면됨.
          2. `Non-parametric` 추정: `사전 정보나 지식 없이 순수하게 관측된 데이터만으로 확률밀도함수를 추정`. 대표적인 예가 `히스토그램`(히스토그램을 구한 뒤 정규화해서 PDF로 사용), `KDE`
  - Kernel 함수란?
    - 원점을 중심으로 대칭이면서 적분값이 1인 non-negative 함수. 가우시안, uniform 함수 등
    - ![스크린샷 2021-08-15 오후 9 56 23](https://user-images.githubusercontent.com/58493928/129513189-8b00fcc8-5c8c-4613-8e2d-387400c99f5e.png)
    - KDE에서는 랜덤 변수 x에 대한 PDF를 다음과 같이 추정함.
      - $\frac{1}{nh}\sum_{i=1}K(\frac{x-x_i}{h})$
      - h는 커널 함수의 bandwidth 파라미터로서 커널이 뾰족한 형태(h가 작은 값)인지 완만한 형태(h가 큰 값)인지를 조절.
        - 1. 관측된 데이터 각각마다(x1, x2..) 해당 데이터 값을 중심으로 하는(x) 커널 함수를 행성한다: K(x-xi)
        - 2. 이렇게 만들어진 커널 함수들을 모두 더한 후 전체 테이터 개수로 나눈다.
      - 히스토그램의 밀도 추정방법과 KDE를 비교하면, `히스토그램은 이산적(discrete)`으로 각 데이터에 대응되는 bin의 값을 증가시켜서 불연속성이 발생하는데, `KDE는 각 데이터를 커널 함수로 대치하고 더함으로써 smooth한 PDF`를 얻을 수 있음. 결과적으로, KDE는 히스토그램의 PDF를 smoothing한 것으로 볼 수 있으며, 이때 `smoothing의 정도`는 `어떤 bandwidth 값`의 `어떤 커널 함수`를 사용했느냐에 따라 달라짐.
        - ![스크린샷 2021-08-15 오후 10 04 50](https://user-images.githubusercontent.com/58493928/129513686-3584f651-83ca-4fbd-a641-95d6e2b26cec.png)

- [x] 모수 추정(Parameter estimation)
  - 모수(parameter): 모집단의 통계량 (ex. 모평균, 모분산..). 관심의 대상이 되는 모집단 특성
    - 분포 곡선을 생성하기 위해 확률 분포 함수(PDF)의 입력 값으로 사용되는 모집단 전체
  - 모집단이 하나 있을 때, 전체 모집단을 측정하는 것이 불가능하기 때문에 해당 모집단에서 랜덤 표본을 추출해서 모집단의 특성(평균/분산)을 추정함.
    >   즉, 추출된 랜덤 `표본`을 사용해서 해당하는 표본 특성을 계산하며, 이 표본 특성이 알 수 없는 모집단 특성에 대한 정보를 요약하는데 사용
    - 해당하는 표본 특성을 `표본 통계량` 또는 `모수 추정치` 라고 함.
    - 통계량은 표본에서 얻은 모수에 대한 정보의 요약이므로 통계량의 값은 모집단에서 추출한 특정 표본에 따라 달라짐 (당연). 표본에 따라 랜덤하게 변하므로, 통계량은 랜덤한 양(변수)임. 
      - 이 `랜덤 변수의 확률 분포`를 `표본 추출 분포` 라고 함.
#
## Machine Learning
#
- [x] Frequentism 와 Bayesian 차이/장점
  - 빈도론(frequentism)
    - 반복되는 사건의 빈도를 다룸.
    - `그 사건이 일어난 횟수의 장기적 비율`.
    - 얼만큼 빈번하게 특정한 사건이 반복되어 발생하는가를 관찰하고 이를 기반으로 가설/모델 검증.
    - 모델의 파라미터가 근본적으로 고정되어 있다는 관점에서 관측결과의 변화를 분석
    - 사건의 빈도, 즉 데이터 관측이 중심. 확률 모델(파라미터)는 고정되어 있고, 주어진 데이터를 가장 잘 설명할 수 있는 모델(파라미터)를 추정한다는 접근.
      - 신뢰구간(confidence level) 95%의 의미: 모집단에서 무한대의 표본을 추출했을때 그 중 95%가 모집단 값을 가지고 있음.
    - `장점`: 여러 번의 실험, 관찰을 통해 알게 된 사건의 확률을 가지고 가설을 검증하는 것이므로 `사건이 독립적이고 반복적이며 그 확률의 분포가 정규분포를 보이는 문제에 잘 맞음`. 대용량 데이터만 처리할 수 있다면 `계산이 복잡하지 않기 때문에 쉽게 처리할 수 있음`.
    - `단점`: 사전에 관찰된 지식이 없는 경우(데이터 부족), 실험 결과의 신뢰가 떨어지는 경우(데이터 결측, 노이즈 포함) 등 `기반이 되는 데이터가 불확실한 경우에는 검증/추론의 질이 낮아짐`.
  - 베이지안(Bayesian)
    - 어떤 가설의 확률을 평가하기 위해 사전지식(사전확률)을 갖추고, 관측결과(데이터)를 기반으로 하는 가능도(likelihood)를 계산해서 앞서 설정한 사전지식을 보정하는 과정을 밟음.
      - $P(H|e) = \frac {P(e|H) * P(H)}{P(e)}$
      - $Posterior(사후확률) = \frac {Likelihood(가능도)*Prior(사전확률)}{Priori(Marginal, 정규화상수)}$
      - `Posterior(사후 확률)`: 직접적으로 관찰 및 표현이 불가능하지만 사후적으로 알 수 있다고...이름을 이따위로 지음. How probable is `hypothesis(or parameter)` given the observed `evidence(관측결과)`
      - `Priori(Marginal, 정규화상수)`: 모델 파라미터와 무관한 관측결과(e) 자체의 확률. How probable is the new `evidence` under all hypothesis
      - `Prior(사전 확률)`: 알고 합리적으로 추정하고 있는 모델 파라미터(H)의 확률. How probable was `hypothesis`  before observing evidence
      - `Likelihood(가능도)`: 모델의 파라미터(H)를 바탕으로 하는 관측결과(e)의 확률. How probable is the `evidence` given the hypothesis
    - `장점`: 관측된 데이터를 가지고 조건부로 가설을 검정하기 때문에 확률 모델이 명확히 설정되어 있다면 베이지안으로 검증된 가설의 타당성은 매우 높다고 여겨짐.
    - `단점`: 현실적으로 사전지식(사전확률, P(H))에 대한 모델링이 어렵고(정확한지 아닌지 판단이 어렵고) 사전지식의 모델링에 따라 결과가 크게 달라질 수 있음.
- [x] 차원의 저주란?
   - 많은 숫자의 잠재적 특성을 고려하게 되면 분류기의 훈련과 수행 속도를 늦출 뿐만 아니라 예측 능력까지 떨어뜨리게 됨. 
- [x] Train, Valid, Test를 나누는 이유는 무엇? 교차 검증(cross validation) 이란?
  - 고정된 test set을 통해 모델의 성능을 검증하고 수정하는 과정을 반복하면, 결국 모델은 test set에만 잘 동작하는 모델이 된다. 즉, test 에 과적합 하게 되므로, 다른 실제 데이터에는 맞지 않음.
  - 따라서 train set을 train + validation set 으로 분리한뒤, validation set을 사용해 검증함.
    - cross validation(교차 검증, validation set 사용해서 검증)의 장/단점
      - 장점
        - 모든 데이터셋을 훈련에 활용할 수 있음. 데이터 부족 방지
        - 모든 테스트셋을 평가에 활용할 수 있음.
      - 단점
        - iteration 횟수가 많기 때문에 모델 훈련/평가 시간이 오래 걸림.
    - k-fold CV(k-겹 교차 검증) vs. stratified k-fold CV (계층별 k-겹 교차 검증)
      - k-fold CV: 데이터가 독립적이고 `동일한 분포를 가진 경우 사용`
        1. 전체 데이터셋을 training/test set으로 나눔
        2. training set을 k 개 폴드로 나눔
        3. k 번째 폴드를 validation set으로 사용, 나머지는 모두 training set으로 사용
        4. k번 반복 후 k개의 평균을 해당 학습 모델의 성능을 취함
      - stratified k-fold CV: 데이터가 `불균일한 분포를 가진 경우(class imbalance)` 사용. k개의 폴드로 나눌때 class 별 샘플 분포 비율을 맞춰서 할당.
- [x] (Super-, Unsuper-, Semi-super)vised learning 이란?
  - Supervised learning(지도학습): 정답 레이블이 있는 데이터셋을 가지고 모델을 학습. ex) classification, regression
  - Unsupervised learning(비지도학습): 정답 레이블이 없는 데이터셋을 가지고 모델 학습. ex) clustering
  - Semi-supervised learning(준지도학습): 정답 레이블이 있는 데이터와 없는 데이터를 동시에 사용해서 더 좋은 모델 빌딩. 레이블이 없는 데이터들의 분포가 만약 균등하다면 지도 학습에 전혀 도움이 되지 않음. 반대로 군집 형태로 나뉜 형태라면 도움!
    - `self-training`: 높은 확률 값이 나오는 데이터에 가중치를 줘서 데이터에 레이블링 함. -- [Self-training with Noisy Student improves ImageNet classification_GoogleBrain+CMU](https://arxiv.org/pdf/1911.04252.pdf)
      1. 레이블이 달린 데이터로 모델을 학습 시킴
      2. 이 모델을 가지로 레이블이 달리지 않은 데이터를 예측함.
      3. 이중에서 가장 확률값이 높은 데이터들만 레이블 데이터로 다시 가져감.
      4. 위 과정을 반복
#
![confusionMatrxiUpdated](https://user-images.githubusercontent.com/58493928/129467249-703212f0-3022-42d8-95ba-31ce128ca27f.jpg)
- [x] Receiver Operating Characteristic Curve 란?
  - 이진 분류 모델의 예측 성능 판단
  - FPR(x-축)이 변할 때, TPR(y-축, Recall)이 어떻게 변하는가.
    - $FPR = \frac {FP}{FP+TN}$
    - $TPR = \frac {TP}{TP+FN}$
  - AUC(Area Under Curve): ROC 곡선 밑의 면적. 1에 가까울 수록 좋다. FPR 작은 상태에서 큰 TPR 얻어야함.
- [x] Precision, Recall, Type I/II error
  - `Precision`: 양성으로 예측한 데이터 중 얼마나 정답? $precision = \frac {TP}{FP+TP}$
    - `Type I error`: `FP`. 양성으로 예측했지만 실제로는 음성. 
    - `FP가 중요한 경우`: 스팸 메일 분류. 실제로는 일반 메일인데 스팸 양성으로 예측해서 처리.
  - `Recall`: 실제 양성인 데이터 중에서 얼마나 정답? $Recall = \frac {TP}{FN+TP}$
    - `Type II error`: `FN`. 음성으로 분류했지만 실제는 양성.
    - `FN이 중요한 경우`: 암 진단/금융 사기. 실제로는 양성인데 음성으로 처리.
  - `F-1 score`: precision과 recall이 어느 한 쪽으로 치우치지 않았을 때 높은 수치. class imbalance 경우에 사용. F-1 = $\frac {2}{\frac {1}{Recall} + \frac {1}{Precision}}$
- [x] Precision Recall Curve?
  - 임계값(threshold)을 어떻게 정하느냐에 따라 precision 과 recall의 trade-off 를 보여줌.
  #
- [Ref1](https://shurain.net/personal-perspective/information-theory/)
- [Ref2](https://sirzzang.github.io/ai/AI-Information-Theory/)
- [x] Information?
  - 잘 일어나지 않는 사건은 자주 발생하는 사건보다 정보량이 많다(informative, 놀라움의 정도)
  - $I(x) = -logP(x)$
    - 수식에서 로그의 밑이 2인 경우는 정보량의 단위를 Shannon 또는 bit 라고 함.
    - 독립인 두 사건의 놀라움은 각각의 놀라움을 더한다고 가정하면, 통신의 관점에서 보면 n개의 선을 사용해서 데이터를 전송하면 한 개의 선을 사용하는 것보다 `n배`의 정보를 전달할 수 있어야 한다는 믿음에 부합함. 
      - 이것을 표현하는 함수는 `로그 형태`임을 Shannon이 발견함. 
    - 반면에 밑이 자연상수(e) 라면 nat 라고 하는데 보통 ML에서는 밑을 e 로 사용함.
- [x] Entropy 란?
  - `정보량의 기대값`. 특정 사건이 일어날 때마다 우리가 얻는 평균적인 정보량(놀라움ㅋㅋ의 양)
  - $p_i$들에 대한 함수. `실제 확률 분포가 p 일 때, 그 확률 분포로부터 얻을 수 있는 정보의 양`
    - $H(x) = \sum_i p_i * log{\frac {1}{p_i}} = -\sum_i p_i * log{p_i}$
- [x] Cross Entropy 와 KL-Divergence 란?
  - `Cross entropy`: 두 개의 확률 분포 p와 q에 대해 하나의 사건 x가 갖는 정보량(서로 다른 두 확률 분포에 대해 같은 사건이 가지는 정보량). `실제 확률 분포가 p일 때, 그와는 다른 어떠한 확률 분포 q로 부터 얻을 수 있는 정보의 양.` 또는, `목표로 하는 최적의 확률분포 p와 이를 근사하려는 확룰분포 q가 얼만 다른지를 측정`하는 방법(즉, 원래 p였던 분포르 q로 표현했을 때 얼마만큼의 비용이 드는지를 측정).
    - Cross Entropy = 최적의 분포 p의 Entropy + KLD
    - $H(p,q) = -\sum_x p(x) * log{q(x)}$
    > ML에서 cross-entropy를 최소화 하는 문제는 결국, `데이터가 보여주는 분포와 모델이 학습한 분포의 각각의 확률이 같아지도록 하는 최적화`
  - `KL divergence`: 분포 `p를 기준으로 q가 얼마나 다른지`를 측정하는 방법. 두 확률 분포 간의 정보량 차이를 나타냄. Cross-entropy는 Entropy 보다 항상 크고, p = q 일 때에만 같으므로 cross-entropy로 부터 entropy를 뺀 값을 두 분포 사이의 거리로 사용. 사실 거리 함수는 아닌데(왜냐면 $D_{KL}(p||q)!=D_{KL}(q||p)$ 니까) 두 분포가 다르면 다를수록 큰 값을 가지며, 둘이 일치할 때에만 0을 갖기 때문에 거리와 비슷한 용도로 사용. 
    - 결국 cross-entropy minimization 문제는 KL divergence를 최소화 하는 문제와 동치임. 즉, q가 p의 분포와 최대한 같아지게 한다는 의미.
      - H(p)는 p의 엔트로피, 즉 우리가 가지고 있는 데이터의 분포이며 학습과정에서 바뀌지 않음. 따라서 q에 대해서 cross-entropy를 최소화 한다는 것은 KL divergence를 최소화 한다는 의미. 따라서 p를 근사하는 q의 확률분포가 최대한 p와 같아질 수 있도록 모델의 파라미터를 조정하는 것.
    - $D_{KL}(p||q) = H(p,q) - H(p) = \sum_x p(x)log{\frac{1}{q(x)}} - p(x)log{\frac{1}{p(x)}} = \sum_x p(x)log{\frac{p(x)}{q(x)}}$
  - Cross Entropy 값은, 예측이 잘못될수록 L1 손실(선형적으로 증가) 보다 더 크게 증가함. 그만큼 더 페널티가 크고 손실 값(loss)이 크기 때문에 학습 면에서도 Cross Entropy를 사용하는 것이 장점이 있음. 그래서 분류 문제에서 자주 사용함.
- [x] Mutual Information 이란?
  - joint probability p(x,y) 와 두 marginal 의 곱(p(x)q(x)) 사이의 KL divergence
  - $MI(X;Y) = D_{KL} (p(x,y)||p(x)q(y))$
  - 두 변수 `X와 Y가 독립`이면 $p(x,y) = p(x)q(y)$ 이므로 두 `분포 사이의 거리가 0 이면(KL divergence가 0이 됨)` 독립일 것이고 `mutual information이 없다`고 볼 수 있다. 반면, 두 분포 사이의 거리가 멀다는 것은(0보다 큰 어떤 값) 결과적으로 두 변수 X와 Y 사이에 공유하는 정보가 많아서 독립이 아니고 mutual information이 발생.
- [x] Cross-Entropy loss 란?
  - `negative` maximum `log` likelihood estimation
    - log 사용 이유:  log는 단조증가함수이므로, log를 취했을 때 최대값을 가지는 지점과 원래 최대값을 가지는 지점이 동일하고, 보통 곱셈보다 덧셈이 계산이 더 간편
  - 예측이 잘못될수록 L1손실(선형적으로 증가)보다 더 크게 값이 증가 --> 페널티가 크고, 학습 면에서도 교차 엔트로피 손실을 사용하는 것이 장점이 있음.
#
  ![스크린샷 2021-08-14 오후 9 17 05](https://user-images.githubusercontent.com/58493928/129466971-cbd85a3a-8322-41e2-9bd5-09b7725eb6fb.png)
- Pattern recognition에서 classification 에 사용하는 모델은 크게 2가지 (Generative model & Discriminative model). 어쨌든 목적은 decision boundary를 구하는 것.
- [x] Decision theory란?
    1. Find parameter by minimizing risk
       - Bayes risk minimization --> Parameters 을 learning. Loss function 을 어떻게 정의하냐에 따라 parameter 값이 결정. 
    2. Decision by Model
       - Parameter를 사용해 posterior를 계산하게 됨.
         - Model에 따라 inference 하는 방법이 달라지는데, `discriminative model`은 parameter를 사용해 directly 하게 posterior를 계산, `generative model`은 joint distribution을 learning 해서 posterior를 indirectly 하게 유추하게 됨.
- [x] Generative model vs. Discriminative model 이란?
  - [분류 문제는 크게 Generative model 과 Discriminative model 로 나눌 수 있음.](https://jayhey.github.io/semi-supervised%20learning/2017/12/08/semisupervised_generative_models/)
  - `Generative model`: 각 `class의 분포에 주목`. Gaussian Mixture Model, Naive Bayes, Restricted Boltzmann Machine. GMM을 예로 들면(clustering 방법 중 하나) 모든 class 들이 Gaussian distribution의 mixture 형태로 주어진다고 가정함. 이 경우는 `데이터와 class의 Joint probability를 계산함`.
    - Likelihood(우도), Posterior probability(사후 확률)을 사용해서 decision boundary(분류 경계선) 만듦. Joint probability(결합 확률) 분포 자체를 찾는 것임.
    - [이상 탐지 기법 중 밀도 기반 추정법들](https://jayhey.github.io/category/#/Novelty%20Detection)(GMM, Mixture of Gaussian Density Estimation, KDE, k-NN based novelty detection, etc.)
  - `Discriminative model`: 두 class가 주어진 경우, 이들 `class의 차이점에 주목`. Logistic regression, LDA, SVM, Boosting, 신경망 등.
    - 직접적으로 모델을 생성.
    - [self-training](https://jayhey.github.io/semi-supervised%20learning/2017/12/07/semisupervised_self_training/)  
#
- [x] 분류와 회귀의 차이
  - 분류(classification): 예측값(결과값)이 `이산형` (ex. 카테고리 같은 것)
  - 회귀(regression): 예측값(결과값)이 `연속형`(ex. 숫자 같은 것)
  - Tree 기반의 회귀
    - 분류: `leaf`에서 예측 결정 값을 만드는데, 특정 `class label 결정`
    - 회귀: `leaf에 속한 데이터 값의 평균 값`을 구해 `회귀 예측 값을 계산`
- [x] 회귀란? (Regression)
  - 여러 개의 독립 변수(x1, x2..)와 한 개의 종속 변수(y) 간의 상관 관계 모델링
    - 결국 최적의 회귀 계수(w)를 찾는 것.
  - $y = w_1 * x1 + w_2 * x2 + ... + w_n * x_n$ 
  - 회귀 계수의 선형/비선형 여부에 따라 '선형 회귀, 비선형 회귀'가 결정됨. (독립 변수 x, 종속 변수 y와는 상관 없음.)
    - 독립 변수 x의 갯수 (1개 - 단일 회귀 simple, 여러개 - 다중/다항 회귀 multiple/polynomial)
      - 다중 회귀 역시 (곡선이지만) 선형 회귀임. 독립 변수들이 2차, 3차 다항식으로 표현될 수 있지만 이를 z1, z2...이렇게 바꾸면 결국 회귀 계수는 그대로 1차, 즉 선형 회귀로 분류됨.
        - 곡선 회귀선이 등장하므로 과대/과소적합 발생 시작 및 Bias-Variance trade-off 등장. 모델이 복잡하면 variance 크고/bias 작고(overfitting, 과대적합), 모델이 단순하면 variance 작고/bias 큼(underfitting, 과소적합). 
    - 종속 변수 y의 갯수 (1개 - 단변량, 회귀 분석 | 2개 - 이변량, 상관 분석 | 여러개 - 다변량, 군집 분석)
      - 변량(variate): 연구자가 각각의 변수에 가중치를 부여해서 변수들의 선형 조합으로 나타낸것. 종속 변수 y 라고 생각해도 되려나..
      - 변수(variable): 연구자가 관찰을 통해 수집한 자료.
  - 선형 회귀: 실제 값과 예측 값 차이(오류의 제곱값, RSS)를 최소화 하는 직선형 회귀선(선형 함수)을 최적화. 규제 방법에 따라서 아래와 같이 나뉨.
    - `일반 선형 회귀`(규제 없음) 
      - 전체 데이터의 오류 합을 구하고 미분 쉽게 하기 위해서 RSS(변수가 w임)를 cost/loss function 으로 사용함. --> 최소화 하는 w값 찾기가 궁극적 목표.
        - cost/loss function 최소화 하는 방법: `경사 하강법(Gradient Descent)` <sub>이건 다른 섹션에서 더 설명.</sub>
        - 하지만 RSS 최소화만 고려하니까 과대적합이 심화됨(회귀 계수가 커짐) 그래서 아래와 같은 규제들이 등장.
    - `Lasso`(L1 규제), 
    - `Ridge` (L2 규제), 
    - `ElasticNet` (L1 + L2 규제), 
    - `Logistic regression` (선형 회귀 방식을 classification에 적용한 알고리즘. 학습을 통해 회귀 최적선을 찾는게 아니라, sigmoid 함수 최적선을 찾고, 이 sigmoid 함수의 반환값(0~1)을 확률로 간주해서 확률에 따라 분류를 함.)
    - 선형 회귀의 경우, `데이터 분포 정규화` & `encoding 방법` 중요
      - `encoding 방법`은 카테고리형의 경우 `one-hot encoding`
      - 선형 회귀 == feature 값과 target 값의 분포가 `정규 분포` 형태를 선호함.
        - 만약, target 값이 왜곡(skew)된 형태의 분포를 가지면 성능 x
        - 주로 `log transformation`을 적용. 다른 scaler 하면 추론 후 원본 값으로 변환이 어려움.
          - 다른 scaler? `StandardScaler (평균 0, 분산 1인 표준 정규 분포 가지게 함)` 또는 `MinMaxScaler (최소값 0, 최대값 1인 분포)`
#
- [X] Overfitting/Underfitting 이란 무엇이고 어떤 문제?
  - 모델의 `복잡도(complexity)`와 연관. 지나치게 복잡하면 over, 지나치게 단순하면 underfitting 발생.
  - 모델의 복잡도 `M` 와 데이터에 대한 복잡도 `D`
    > - Overfitting: 학습 데이터에 너무 적합하게 훈련되어 있어서 검증/테스트 데이터에서 loss가 증가하는 것. 학습 데이터에 대한 성능은 좋지만, 테스트 데이터에 대한 성능이 나쁜 경우. M > D
    > - Underfitting: 학습 데이터에 대해서 충분히 학습하지 못한 경우. 학습 데이터에 대한 성능이 테스트 데이터에 대한 성능 보다 좋지 못한 경우. D > M
    > - 일반화(Generalization): 모델이 보지 못한 Test 데이터에 대해서 예측을 잘하는 경우, '일반화가 잘 되었다'라고 표현함.
- [x] Overfitting 과 Underfitting 을 해결하는 방법은?
    - Overfitting 을 해결하는 방법 3가지
      1. 데이터의 수(D)를 늘린다.
            
            > 학습이 경과됨에 따라서 학습 데이터를 모두 학습한 상태일 경우, 데이터에 대한 패턴을 배우는 것이 아니라 데이터를 기억하는 수준이 됨. 패턴 예측을 못함. 따라서 D를 증가시켜준다. 데이터 증강법 사용.
      2. 모델의 Complexity를 (M) 줄인다.
            
            > 모델이 복잡하면 데이터에 fitting 하기 위해서 모델의 계수가 증가하게 됨.
      3. 규제(Regularization)을 사용한다. 
            
            > 앞서와 같은 이유로 계수의 크기를 줄여주기 위한 방법 사용. Lasso/Ridge 등.
    - Underfitting 을 해결하는 방법
      1. 파라미터가 더 많은 강력한 모델을 선택한다.
      2. 학습 알고리즘에 더 좋은 특성을 제공한다.
      3. 모델의 제약을 줄인다.
- [x] Regularization 이란?
  - 모델의 복잡도를 줄이기 위한 방법.
  - 일반적인 선형 회귀의 overfitting 문제를 해결하기 위해 '회귀 계수'에 페널티 값을 적용.
    - cost function에 alpha 값으로 페널티를 부여해 회귀 계수 값을 감소시켜 과적합 개선
    > cost function 최적화 = Min$(RSS(w) + \alpha * |w|_i^i)$
    - $\alpha$ 값을 크게 하면, cost function은 회귀 계수(w)를 작게 해서 과적합 개선
    - $\alpha$ 값을 작게 하면, w 값이 커져도 어느 정도 상쇄가 가능하므로 과소적합 개선 (하지만 $\alpha$ 값을 0부터 크게 하는 방향으로 테스트, 회귀 계수를 감소 시키는게 보통임.)
- [x] Lasso/Ridge/ElasticNet 이란?
  - `Lasso` (L1): $\alpha * |w|_1^1$. w의 절대값에 대해 페널티 부여. 영향력 크지 않은 `회귀 계수 값을 0으로 만들어서 제거`함. 중요한 feature만 선택할 수 있음(feature selection).
  - `Ridge` (L2): $\alpha * |w|_2^2$. w의 제곱에 대해 페널티 부여. 회귀 계수를 0으로 만들진 않음. 대신 상대적으로 큰 '회귀 계수' 값의 예측 영향도를 감소시킴. 즉, 회귀 계수 값을 더 작게 만든다(`회귀 계수 조정`).
  - `ElasticNet` (L1 + L2): feature가 많을때 사용. L1으로 feature 개수를 줄이고 L2로 계수 값의 크기 조정. Lasso 에서 중요한 feature 만 선택하고 나머지 회귀 계수는 0으로 만들어서 $\alpha$ 값에 따라 회귀 계수 값이 급격히 변동할 수 있기 때문에, 이를 완화하기 위해 사용함. 하지만 수행 시간 오래 걸림.
  #
- [x] Regularization_version2 
  - Deep Learning_Ian Goodfellow_Ch7. Regularization for deep learning
  - 학습 데이터 뿐만 아니라 새로운/테스트 데이터도 잘 맞춰야 함. 즉, 오버피팅 해결해야..!
  - 책에서 말하는 '정규화'는 training error를 줄이는게 아니라 `generalization error (training error - test error)를 감소시키는 것`. 여태 내가 알고 있는 '정규화란 모델의 복잡도를 줄이는것' 이라는 개념을 넘어서 generalization error를 줄이기 위한 모든 방법 총망라.
    - > We defined regularization as "any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error."
    1. `Parameter Norm Penalities` (내가 정규화라고 알고 있던 것)
         - 위 설명 참조.
    2. `Dataset Augmentation`
        - 학습 데이터의 크기를 늘리는 방법
    3. `Noise Robustness`
        - > 머신러닝에서의 Robust? : 이상치나 노이즈가 들어와도 크게 흔들리지 않아야(robustness) 함.
        - 학습 단계에서 일부러 노이즈를 주는 방법.
          - 레이어 중간에 noise injection(노이즈 추가), 또는
          - classification 할 경우, `라벨을 부드럽게(label-smoothing)` (예: 1,0,0 -> 0.8, 0.1, 0.1)
    4. `Semi-supervised learning`
        - `self-training`: 라벨이 있는 데이터로 모델을 학습 후, 라벨이 없는 데이터에 대해 예측해서 (예측) 확률이 높은 데이터에 라벨링을 하고 학습 데이터로 포함시킴. 노이즈를 추가한다고도 할 수 있음.
    5. `Multi-task learning`
        - 서로 다른 문제(task) 속에서 몇 가지 공통된 중요한 요인(factor)이 뽑히며, shared 구조를 통해서 representation(feature extraction은 결국 representation을 찾는 것)을 찾을 수 있음. 모델의 앞단에 있는 shared 구조 덕분에, 각각의 요인을 학습시킬 때보다 더 좋은 성능을 낸다.
    6. `Early stopping`
        - training loss는 줄어들지만 validation loss 는 증가하면(오버피팅이 발생하면) 학습을 멈추는 것.
    7.  `Ensemble`
    8.  `Dropout`
        - 학습할 때 뉴런의 일부를 네트워크 상에서 비활성화 시킴. (테스트할 때는 모든 뉴런 사용)
        - dropout을 통해 앙상블 학습 처럼 마치 여러 모델을 학습시킨 효과를 줌.
        <!-- 11.  `Adversarial training`
        - 사람이 관측할 수 없을 정도의 작은 noise를 넣으면, 완전 다른 클래스가 나옴. 입력은 아주 조금 바뀌었지만, 출력이 매우 달라짐. 분류 경계선 근처에 있는 데이터 값을 조작하는 경우를 생각해보자. 역전파를 통해 신경망의 가중치를 조정하는 것이 아닌, 입력된 데이터(이미지)를 아주 조금씩 변경하며 정답 값에 가깝게 만듦. -->
#
- [x] Activation function 이란? 3가지 activation function type
  - 존재이유: 선형성을 가지는 딥러닝 네트워크에 `비선형성`을 적용하기 위함. 선형 활성화 함수는 선형 레벨의 판별 기준 제공 / 비선현 활성화 함수는 복잡한 함수를 근사할 수 있게함. 대신 필연적으로 과적합을 피할 수 없게됨.
    - 이전 perceptron 에서는 입력층에서 전달된 정보는 가중치가 곱해지고 합산된 값이 편향 보다 크거나 작다에 의해 정보가 전달(0 또는 1) 됐음. 즉, 임계값을 넘으면 정보가 전달되고, 임계값을 넘지 않으면 정보가 전달되지 않는 식의 step-function 이었음.
   1. `Ridge activation function`: input 변수들의 선형 조합에 대한 다변량 함수 적용 (그냥 말 뜻 자체가 `Heaviside` 의미)
      - Linear, ReLU, Heaviside, Logistic(sigmoid)
        - `Linear`
          - 비선형성을 더할 수 없음. 그냥 연산 값이 커질뿐임. 또한 입력 값에 이상치가 존재하는 경우, 분류를 불가능하게 만든다.
        - `sigmoid`
          - $f(x) = \frac {1} {(1 + e^{-x})}$ , $f'(x) = f(x)*(1-f(x))$. 
          - 입력 값을 0과 1 사이의 값으로 변환하여 출력함. Logistic regression/Binary classification 에 사용. 보통 출력층에서만 사용됨. 은닉층에 사용하면 아래와 같은 한계점 때문에 성능 x
          - ![스크린샷 2021-08-13 오전 10 08 14](https://user-images.githubusercontent.com/58493928/129395034-8dd79a15-16d7-4c1c-ae3c-bf1bc2e2e332.png) 
          - `한계점` : `출력값이 너무 작아 제대로 학습이 안되고`, `학습 속도가 느림`
            - (1) 출력 값의 평균이 0이 아니고 출력 값이 모두 양수기 때문에 GD할때 기울기가 모두 양수이거나 음수가 됨. 따라서 `gradient update가 zig-zag로 발생`하며 학습 속도가 저하됨. 
            - (2) 아무리 큰 값이 입력되도 0~1 사이의 값만 반환하므로, 값이 일정 비율로 줄어든다. 또한 출력 값의 중앙 값이 0.5 이며 모두 positive 이기 때문에 출력의 가중치 합이 입력의 가중치 합보다 커지게 됨. (편향 이동 `bias gradient` 발생) 즉, 신호가 각 레이어를 통과할 때마다 분산이 계속 커지게 됨. --> (3) 연결
            - (3) 입력 값이 커지거나/작아지면 (레이어가 깊어질수록) 미분 값이 0에 수렴해서 역전파시 0 이 곱해지고 gradient가 전달되지 않는 `vanishing gradient` 발생 --> `tanh`가 나와서 어느 정도 개선했지만(평균이 0) 큰 입력 값에 대해서 미분 값이 0에 수렴하는 현상은 계속 발생 --> ReLU 등장
        - `ReLU`(rectified linear unit)
          - $f(x) = x (x>=0)$ or 0 $(x<0)$ , $f'(x) = 1 (x>=0)$ or $0 (x<0)$
          - ![스크린샷 2021-08-13 오전 10 16 08](https://user-images.githubusercontent.com/58493928/129395932-f15555d8-d9ac-43dc-813d-727a7202c0ad.png)
          - 0 미만의 값은 0. 
          - 전파되는 값들이 크고(0이상일때 그대로 전달) 역전파 되는 값들 역시 (y=x를 미분하면 1이 나오기 때문에) 기울기 값이 그대로 전파되므로 학습 속도가 빠름. 또한, 연산과정에서 sigmoid/tanh 는 지수연산이 필요하지만 ReLU는 값을 그대로 전달해주기 때문에 속도가 빠름.
          - `한계점`: 한 번이라도 출력 값이 0 미만의 값이 다음 레이어에 전달되면 이후의 뉴런들의 출력값이 모두 0이 되는 `dying ReLU` 발생. 따라서 음수 출력 값을 소량이나마 다음 레이어에 전달하는 방식으로 개선한 ReLU 변종 함수들 등장.
   2. `Radial activation function`
      - RBF(Radial Basis Function) network에서 사용됨. 효율이 매우 좋은 universal function approximator (). 
        - RBF는 training set에 대한 input의 유사도를 측정함. 
   3. `Folding activation function`
      - CNN의 pooling layers 또는 multiclass classification network의 아웃풋 레이어에서 광범위하게 사용됨.
      - mean, minimum, or maximum 등을 인풋에 대해서 취함. Multiclass classification 에서는 `softmax` 가 대표적
        - `softmax`: 분류될 클래스가 n개 일때, n차원의 벡터를 입력받아서 각 클래스에 속할 확률을 추정함.
          - 지수함수를 사용하면, 아무리 작은 값의 차이라도 확실히 구별될 정도로 커짐. 또한 미분이 쉽다^^
          - ![스크린샷 2021-08-13 오전 10 39 38](https://user-images.githubusercontent.com/58493928/129398337-0d2e342b-362b-4643-932f-c781ec5b4d2e.png)
- [x] CNN 에 대해 설명
  - 특정 공간 영역(receptive field)에 대해서 low->high level로 추상화 시킴. universal feature extraction.
  - `필터`(여러개의 커널로 구성)(정방행렬)을 원본 이미지에 순차적으로 슬라이딩 해가면서 새로운 픽셀값을 만든다. (= 합성곱. Convolution). 즉, 이미지를 `이루고 있는 픽셀 배열을 변경`하여 이미지를 변경함.
    - Filter >> Kernels
    - `합성곱`: 필터에 mapping 되는 원본 이미지 배열과 필터 배열을 element-wise 하게(1:1 매칭) 곱셈을 적용한 뒤 합을 구함
    - `커널 사이즈`: 보통 정방 행렬. 커널이 크면 클 수록 입력 feature map(convolution을 적용한 결과물)에서 더 큰/더 많은 feature 정보를 가져올 수 있으나, 많은 파라미터&연산량을 요구함.
    - 개별 커널은 필터 내에서 다른 값을 가질 수 있고, 학습을 통해 스스로 최적화 함.
    - `stride`: 슬라이딩 윈도우(필터)가 이동하는 간격. stride가 크면 공간적인 특성(spatial feature)을 손실할 가능성이 높아지지만, 꼭 중요한 feature 손실을 의미하진 않음. 오히려 불필요한 feature를 제거하는 효과를 가져올 수 있고 그래서 convolution 연산 속도 향상이 가능함.
    - `pooling`: feature map의 일정 영역 별로 하나의 값을 추출하는 subsampling. 비슷한 feature map 들이 서로 다른 이미지에서 '위치'가 달라지면서 다르게 해석되는 현상이 중화됨. 하지만 특정 위치의 feature 값이 손실될 위험이 있음. 그래서 stride로 대체하는 경향도 있음.
        
        > Convolution 연산 진행하면서 점차 `feature map 크기를 줄이면`, 위치 변화에 따른 feature 값 영향도(spatial invariance)를 줄일 수 있고, 이는 generalization/overfitting 감소 등의 장점으로 연결.
    - `padding`: 합성곱 연산 수행시 출력 feature map이 입력 feature map 보다 계속 작아지는 것을 막음. 모서리 feature 잡아냄. 입력 이미지에서 충분한 특성 추출.
      - `same`: 아웃풋이 인풋과 동일한 길이를 갖도록 인풋 패딩
      - `valid`: 패딩 없음
      - `causal`: 입력의 왼쪽에 0을 알맞게 패딩하고 valid 처리. 시계열 모델에서 미래 시퀀스 보지 않기 위함.
    - `feature map size = floor(((I-K + 2P)/S) + 1), I = 이미지 크기, K = 필터 크기, S = 스트라이드, P = 패딩 크기`
    - `dropout`: 연결망의 무작위 drop(연결 줄임) --> overfitting 개선. 
    - `GAP`: Flatten의 여러 이슈들(3차원 feature map을 1차원으로 줄여서 FC layer에 연결하면 파라미터 숫자가 급증해서 연산량 증가하고 overfitting 발생) 개선 위해 channel 별로 (각 feature map 별로) average 값 1개씩 뽑음. 따라서 spatial information 을 유지하면서 category로 직접 연관 시킬 수 있음. 별로의 parameter optimization 필요하지 않아서 연산량 적고 overfitting도 차단.
- [x] RNN 에 대해 설명
  - 각 타임 스텝(time step) 마다 `입력(x(t))`과 `이전 타임 스텝(y(t-1))`을 입력으로 받음. 따라서 2개의 가중치를 가짐.
  - `BPTT(BackPropagationThroughTime)`: RNN 훈련. 일련의 타임 스텝 진행하고나서 gradient가 전파됨. 즉, 정방향 패스 동안에는 모두 동일한 가중치 적용. 
  - 입력과 출력 시퀀스에 따른 RNN 분류
    1. Sequence-to-Sequence
         - 주식 가격 같은 시계열 데이터 예측
         - 입력 시퀀스에 따른 출력 시퀀스 발생
    2. Sequence-to-Vector
        - 영화 리뷰에 있는 연속된 단어를 주입하면 감성 점수 출력
        - 입력 시퀀스 주입하고, 마지막을 제외한 모든 출력 무시
    3. Vector-to-Sequence
        - 이미지를 입력해서 이미지에 대한 캡션 출력
        - 각 타임 스텝에서 하나의 입력 벡터를 반복해서 네트워크에 주입하고 하나의 시퀀스 출력
    4. Encoder-Decoder
        - 번역
        - encoder(seq-to-vec) + decoder(vec-to-seq)
   - 긴 시퀀스 다룰 때, 많은 타임스텝에 걸쳐 실행해야 하므로 펼친 RNN이 매우 깊은 네트워크가 됨. 그래서 `gradient vanishing/exploding` 또는 `훈련 시간이 매우 길거나 불안정` or `입력의 첫 부분을 잊어버림(Long-term memory loss, 단기 기억 문제)`
    1. `Gradient vanishing/exploding (unstable gradient) 문제`
        - `원인`: tanh 가 기본 활성화 함수(= 수렴하는 활성화 함수 사용): 동일한 가중치가 모든 타임스텝에서 사용되기 때문. 그래서 증가-->계속 증가 // 감소 --> 계속 감소
        - 해결책1: `LayerNormalization`
          - 특성 차원에 대해 정규화
          - 입력마다 하나의 스케일과 이동 파라미터를 학습(BN과 비슷)
          - 샘플에 독립적으로 타임스텝마다 동적으로 필요한 통계를 계산할 수 있음. 즉, 훈련과 테스트에서 동일한 방식으로 작동함 (BN과 다름)
        - 해결책2: `dropout` or `recurrent_dropout` 매개변수 사용
          - dropout: (타임스텝마다) 입력에 적용하는 드롭아웃 비율 정의
          - recurrent_dropout: (타임스텝마다) 은닉 상태에 대한 트롭아웃 비율 정의
        - 해결책3: `Gradient clipping`
          - 기울기 폭주를 막기 위해 임계값을 넘지 않도록 기울기 값을 자름.(임계치 만큼 크기를 감소 시킴.)
          - RNN에서 BPTT에서 시점을 역행하면서 기울기를 구하는데, 이때 기울기가 너무 커질 수 있으므로 사용.
          
    2. `단기 기억 문제(Long-term memory loss)`
        - RNN을 거치면서 데이터가 변환되므로 일부 정보는 매 훈련 스텝 후 사라짐. 따라서 어느정도 시간이 지나면 RNN의 상태는 (거의) 첫 번째 입력의 흔적 없음.
        - 해결책1: `LSTM(Long-Short-Term Memory)`
          - LSTM 셀은 중요한 입력을 인식하고(입력 게이트), 장기 상태에 저장하고/필요한 기간 동안 보존하고 (삭제 게이트), 필요할 때마다 이를 추출하기 위해 학습함(출력 게이트).
          - 네트워크가 `입력/삭제/출력 게이트를 가지고 장기 상태에 저장할 것, 버릴 것, 읽어들일 것을 학습`함.
        - 해결책2: `GRU cell`(Gate Recurrent Unit)
          - `출력 게이트 없음`. 즉, 전체 상태 벡터가 매 타임스텝 마다 출력됨. 그러나 이전 상태의 어느 부분이 주 층에 노출될지 제어하는 새로운 `게이트 제어기 있음`.
          - `게이트 제어기`가 삭제 게이트와 입력 게이트를 모두 제어함. 기억이 저장 될때 마다 저장될 위치가 먼저 삭제됨.
        - 해결책3: `Conv1D`
          - 커널이 시퀀스 위를 슬라이딩 하며 지나가고, 커널마다 1D feature map 출력함. 
          - 각 커널은 매우 짧은 하나의 순차 패턴을 감지하도록 학습됨
         - 해결책4: `WaveNet`
           - 층 마다 (각 뉴런의 입력이 떨어져 있는 간격인) 팽창 비율(dilation rate)을 2배로 늘리는 Conv1D 층을 쌓음.
             - 하위 층은 단기 패턴, 상위 층은 장기 패턴을 학습함. --> 매우 긴 시퀀스 잘 처리
             - padding='causal' 사용. (입력의 왼쪽에 0을 알맞게 패딩하고 'valid' 처리해서 합성곱 층이 예측을 만들 때 미래의 시퀀스를 훔쳐보지 않음.)
         - 해결책5: `TabNet` 은 패스. 지금 캐글 참여 중.

- [x] Newton's method 이란 무엇?
  - [Newton's method](https://darkpgmr.tistory.com/58?category=761008)
  - 상당한 고차 방정식 $f(x) = 0$ 의 해를 근사적으로 찾을 때 사용
  - $f'(x)$가 $x=a$ 에서의 접선의 기울기라는 점을 이용함. 
  - 가령,
    - 아무 값이나($x=a$) f에 넣고 $f(a)$의 값을 살펴본다. 만일 $f(a) > 0$ 이고, $f'(a) > 0$ 이면, $f(x) = 0$ 이 되는 x는 a 보다 작은 값이다. (2차 함수 그려서 생각해보면 간단하다)
    - 하지만 얼마나 값을 줄여 나가야 할까? 만약 $|f(a)|$ 이 작고, 접선의 기울기가 가파르다면(크다면) 바로 근처에 해가 있을 것이고, 반대로 $|f(a)|$가 크고 접선의 기울기가 완만하다면(작다면) 멀리 떨어진 곳에 해가 존재할 것이다.
    - 뉴턴법/뉴턴-랩슨법은 `현재 x 값에서 접선을 그리고 접선이 x축과 만나는 지점으로 x를 이동시켜가면서 점진적으로 해를 찾는 방법`이다.
      - $x^{t+1} = x^{t} -  \frac {f(x^t)} {f'(x^t)}$
      - 종료 조건은 x 값의 변화가 거의 없을 때까지. 즉 $|x^{t+1} - x^t|$ 가 매우 작은 값이면 뉴턴법을 종료하고 $x=x^{t+1}$이 solution 이라고 생각하는 것.
    - 제약 조건: f(x)가 연속이고 미분가능해야함. 해가 여러 개라면 그 중 하나의 해를 찾아줄 뿐. 만약 해가 여러 개인 경우, 초기값 x1을 어떻게 주느냐에 따라서 뉴턴법으로 찾아지는 해가 달라질 수 있음. 따라서 일정한 간격으로 x값을 변화시키면서 함수 값의 변화를 지켜본 후, 함수 값의 부호가 바뀌는 구간마다 interpolation 으로 초기값을 설정함.
    - 활용
      1. $f(x) = 0$ 의 해를 구할 때
      2. 서로 다른 두 함수 g(x)와 h(x)의 값이 같게 되는 x를 구할 때 ($f(x) = g(x) - h(x)$로 놓고 $f(x)=0$인 x를 구함)
      3. f(x)의 최소값/최대값을 구할 때
          - 일반적으로 함수는 극점에서 최대/최소값을 가지므로 f'(x)=0 인 x를 뉴턴법으로 구한 후 f(x)에 대입하면 f(x)의 최대값/최소값을 구할 수 있음. 

- [x] 경사하강법(Gradient Descent)
  - 미분을 통해 비용 함수(cost function)의 극소점(local minimum)을 찾는 방법 중 하나.
  - 오차(loss)를 개별 가중치(w 변수)에 대해 편미분 해서 구해지는 기울기(gradient)와 학습률(learning rate, $\eta$) 활용.
    - 목적지(최소 오차, 비용 함수의 최소점)로 가는 방향(기울기)으로 정해진 보폭(학습률)만큼 가중치 값을 수정해서 오차를 줄여나감. 
    - $w_{1,new} = w_{1,old} - \eta * (\frac {dLoss}{w1})$
  - 미분 값이 계속 감소하는 방향으로 순차적으로 w를 업데이트 함. 미분된 1차 함수의 기울기가 감소하지 않는 지점을 비용 함수가 최소인 지점으로 간주하고 그때의 w를 반환.
    - 경사하강법의 주요 문제
      - `learning rate(step)`($\eta$) 크기에 따른 이슈: 너무 작으면 최소점에 수렴하는데 오랜 시간 걸리고, 너무 크면 최소점 찾지 못하거나 오히려 발산
      - `global minimum과 local minimum` 이슈: 매우 많은 파라미터를 가진 복잡한 함수에는 local minimum이 여러개 있음(saddle point 포함). 어떤 특정한 local min.에 갇혀버릴 수 있음 --> `optimizer`등장
  1. `GD`: 전체 학습 데이터에 대해서 한꺼번에 로드하고 GD 계산. 자원 소모 심함.
  2. 확률적 경사하강법(Stochastic Gradient Descent)(`SGD`): 전체 학습 데이터에 대해서 임의로 한 건만 선택해서 GD 계산. 비교적 잘 맞지만 zig-zag 움직이며 oscillation 이 큼. 수렴이 조금 어려움.
  3. 미니배치 경사하강법(`Mini-batch SGD`): 전체 학습 데이터에 대해서 `특정 크기 만큼(batch 크기)` 임의로 선택해서 GD 계산.
       - `epoch`: 모델이 주어진 훈련 데이터 모두 소진하는 구간
       - `batch`: 가중치를 1번 업데이트 하는데 사용하는 데이터 집합 (즉, 훈련 한 번에 사용되는 데이터 수)
       - `iteration`: epoch당 훈련 횟수
         - 예를 들어, 훈련 데이터셋 12개에 대해서(1epoch) batch 크기를 2로 정하면(batch_size=2) iteration은 6.
       - `batch 크기가 엄청 작으면`: (예를 들어 크기 1 ==> SGD) `각각의 데이터를 상대로 학습이 꼼꼼하게 이루어짐`. w 자주 업데이트. 하지만 `전체 훈련 데이터의 '분포'에서 비정상적으로 떨어진 '특이값(outlier)'에 따라 가중치 업데이트에 큰 편차가 발생할 수 있음(zig-zag).` 특이값 때문에 구해지는 gradient가 실제 최적 탐색 경로를 크게 벗어남.
       - `batch 크기가 엄청 크면`: (크기 == 데이터 수) `주어진 모든 훈련 데이터의 평균 특성을 파악`함으로써 한 번의 훈련으로 최적의 가중치 조합을 찾음. 속도 빠름. `하지만!! 모델 최적화(optimization) 및 일반화(generalization) 어려움`
         - `모델 최적화가 어렵다?`: `오차 최소화하는 가중치 찾기가 어렵다`는 말임. local minimum이나 saddle point에 빠져서 학습 중단.
         - `모델 일반화가 어렵다?`: 학습 데이터와 조금이라도 다른 성격의 데이터를 받으면 모델이 제대로 동작하지 않음.
         - 따라서 `적당한 배치 크기 탐색`중요(mini-batch GD).
              
              > adaptive learning rate 도입 대두. 즉, 큰 배치에서의 학습이 작은 배치 때와 유사해야함. 층 별로 학습률 조정. --> `optimizer` 개선함.
  
- [x] 역전파(Backpropagation)
  - 보다 복잡한 문제의 해결을 위해 입력층과 출력층 사이에 여러 은닉층(hidden layer)를 넣으면 파라미터수가 급증해서(왜냐하면 입력값 * 가중치를 합치고 활성화 함수에 넣는 과정 반복) 미분이 거의 불가능해짐 --> 역전파 등장, 복잡한 심층 신경망 학습.
  - 연쇄법칙(chain rule) 통해 기울기(gradient)를 구하고, 그 기울기에 학습률($\eta$)을 곱한 값을 기존 가중치에서 빼서 가중치를 업데이트 함.
  - 1. Feed forward(순전파) 수행: 데이터를 입력 받은 모델은 (초기화된) 가중치를 이용해 예측값을 출력. 즉, 예측값을 구하는 과정.
  - 2. 손실함수를 통해 오차(loss, 예측값-정답) 파악
  - 3. 출력층에서 입력층으로 역순으로 거슬러 올라가며 (손실함수 그래프에서 오차를 최소화 하는 가중치 탐색, == 최적화 함수는) 얻은 gradient를 이용해 앞서 구한 오차를 최소화 하는 방향으로 (출력층과 가까운 층 부터) weight 값을 업데이트 함. 
  
- [x] Optimizer 의 종류와 차이
  - 최소 loss로 보다 빠르고 안정적으로 수렴할 수 있는 기법 적용. 즉, `빠르고 안정적으로 global min. 수렴`
  - `실제로 GD로 w를 업데이트 하는 알고리즘`.
    - `Momentum` 방식: Momentum, NAG. 가속도를 준다. 이를 통해 plateau 와 local minima를 뛰어 넘어 일반적인 global minimum에 도달할 수 있다. 일반적인 GD 계산에 momentum 계수 도입. 
    - `Adaptive` 방식: Adagrad, Adadelta, RMSProp. 방향을 최대한 일직선으로. 가중치 별로 서로 다른 learning rate을 동적으로 적용. 그동안 적게 변화된 가중치는 보다 큰 learning rate을 적용, 반대는 반대로..한다든지, 지수 가중 평균 계수(exponentially weighted average) 도입 등.
    - `Momentum + Adaptive` 방식: Adam. RMSProp과 같이 개별 weight에 서로 다른 learning rate을 적용함과 동시에 gradient에 momentum 효과를 부여해서 과거 gradient 영향을 감소시키도록 지수 가중 평균 적용함. 즉 weight 마다 update 변화량을 다르게 줘서 zig-zag 를 막음. 하지만 최근은 adaptive 방식에 대한 의구심으로 adagrad로 회귀중..?
      - `RAdam + LookAhead`: 초반 warmup을 잘하는 것 + 도우미 ==> k번 업데이트 후 처음 시작했던 방향으로 1step back 후 그 지점에서 다시 k번 업데이트. local minima를 빠져나올 수 있음. (k steps forward + 1 step back)

- [x] Local optimum 으로 빠지는데 성능이 좋은 이유는 무엇?
  - (정답인지 모르겠음) 실제 딥러닝에서 local optima 로 빠질 확률이 거의 없음. 실제 딥러닝 모델은 w가 수도 없이 많으며, 그 수 많은 w가 모두 local optima에 빠져야 w 업데이트가 정지 되기 때문. 이론적으로 불가능.
  - 진짜 문제는 `Plateau`가 더 자주 빈번하게 발생하고 loss 업데이트 멈춤.

- [x] Internal Covariate Shift
  - 일단, 학습 데이터셋과 테스트 데이터셋의 분포 차이(Covariate Shift)는 모델의 성능 저하의 주요 원인임. (주로 학습한 것과 전혀 다른 것이 들어오면 못맞추는 것)
  - 이처럼, 학습/테스트 데이터셋 간의 차이에 대한 문제(Covariate Shift)를 
    1. 각 mini batch간 input 데이터의 차이에 의한 문제로 확장 시킨 것을 `Internal Covariate Shift` 라고 함.신경망에서는 보통 모든 학습 데이터를 한 번에 사용하지 않고 mini batch를 사용하는데, 각 step에서 사용되는 학습 데이터는 매번 달라지게 됨. 이렇게 배치 간의 분포가 다른 경우를 `internal covariate shift` 라고 함.
    2. 신경망 각 층을 통과할 때마다 입력 데이터의 분포가 조금씩 변경/누적되는 현상 발생. 
       - 왜냐하면, 각 Layer 마다 input을 받아서 linear combination 을 구한 후 activation function을 적용해서 output을 구하는 작업이 이루어지기 때문. 결과적으로 이 때문에 각 layer의 input data 분포가 달라지게 되며, 뒷단에 위치한 layer 일 수록 변형이 누적되어 input data 의 분포가 상당히 많이 달라지게 됨.
       - 따라서 모델들의 parameter 들이 일관적인 학습을 하기가 어려워짐.
    - ![스크린샷 2021-08-13 오전 12 01 55](https://user-images.githubusercontent.com/58493928/129317781-5cc57360-9d30-4d1e-a951-4dadbede89ab.png)
  - 게다가 weight가 특이하게 큰 것이 있으면 영향력이 커짐.
    - 그래서 각 층을 거칠 때마다 z-score로 다시 scaling 함 ==> `BatchNormalization`
- [x] Batch Normalization 은 무엇이고 왜 하는지?
  - `배치 마다 정규화`
  - `Regularization` 효과: 예를 들면, 2D 이미지에서 i는 $(i_N, i_C, i_H, i_W)$이고 4개의 벡터(N, C, H, W)를 갖게 된다. 배치 정규화에서는 미니배치 크기(N)에 대해 각각의 특성(C, H, W)의 평균과 표준편차를 이용해 정규화한다. 
  - `Internal Covariate Shift` 해소 하기 위해.
  - 각 층을 거칠 때마다 z-score로 다시 scaling 함. 
    - weight 자체가 작아지고 변동성이 작기 떄문에 GD가 빠르게 수렴할 수 있음.
    - Batch 마다 하며, scaling 파라미터와 shift 파라미터는 학습으로 동적 갱신함.
    - 테스트 데이터에 대한 BN은 학습 때와 다름.
      - 해당 mini-batch에 대한 표본 평균과 표본 분산을 쓰는 것이 아니라(각 mini-batch 마다 표본 평균과 표본 분산이 다르니까 normalization 결과가 달라지잖니..), 학습 과정의 K개의 mini-batch에서 얻은 K개의 표본 평균을 평균낸 값(learning mean)과 K개의 표본 분산을 평균낸 값(learning variance)을 사용함.
- [x] BatchNormalization vs. LayerNormalization
  - `mini-batch`: 동일한 feature 개수들을 가진 다수의 샘플들
  - BatchNormalization
    - 각 층에서 활성호 ㅏ함수를 통과하기 전이나 후에 모델에 연산을 추가
    - 입력을 원점에 맞추고 정규화한 다음, 각 층에서 두 개의 새로운 파라미터로(scale, shift) 결과값의 스케일을 조정하고 이동
    - ![스크린샷 2021-08-24 오후 11 59 10](https://user-images.githubusercontent.com/58493928/130742014-918a1ff7-561a-435f-8566-85ec3a3eb595.png)
  - LayerNormalization
    - ![스크린샷 2021-08-24 오후 11 59 24](https://user-images.githubusercontent.com/58493928/130742072-8e4308ff-cc01-442a-b5c6-8b141579df08.png)

#
- [x] Ensemble 이란?
  - 여러 개의 classifier 예측 결합 
  - Use multiple learning algorithms to obtain the better performance
- [x] Stacking 이란?
  - 여러 가지 다른 모델의 예측 결과 값을 다시 학습 데이터로 만들어서 다른 모델(meta 모델)로 재학습 시켜서 결과 예측
  - Train additional merger model
- [x] Bagging 이란?
  - 복잡한 결정 트리 모델을 여러개 조합해서 분산(variance)를 줄임.
  - Ensemble vote with equal weight (reduce variance)
  - 각각의 classifier가 모두 같은 유형의 알고리즘 이지만, 데이터 샘플링을 다르게 하면서(bootstrapping - 데이터 중첩 허용. 허용 하지 않는 것은 pasting) 학습 수행 후 voting.
    - oob 평가(out-of-bag): bagging 에서 선택되지 않은 훈련 샘플 나머지에 대해서 테스트 함.
    - 랜덤 패치(random patch): 훈련 feature 와 sample을 모두 샘플링
    - 랜덤 서브스페이스(random subspace): 훈련 샘플 모두 사용 하고 feature는 샘플링 --> 더 다양한 classifier를 만들어서 bias 증가, variance 감소
- [x] Boosting 이란?
  - 간단한 모델을 여러개 조합하여 bias(편향) 줄임.
  - A set of weak learner --> strong learner (reduce bias)
  - 여러 개의 weak classifier (오분류율이 0.5를 초과) 가 순차적으로 학습을 수행하되, 앞에서 학습한 classifier가 예측이 틀린 데이터에 대해서는 올바르게 예측할 수 있도록 다음 classifier에는 가중치(weight)를 부여하여(boosting) 학습
- [x] Bagging 과 Boosting 차이?
  - Bagging: 같은 classifier 알고리즘(알고리즘만 같을 뿐) + 다른 데이터 샘플링(중첩 허용. 허용 하지 않으려면 pasting) + voting
  - Boosting: 같은 classifier `순차적` 학습 + 앞서 학습한 classifier 예측이 틀리면 다음 classifier에 가중치(weight) 부여
- [x] AdaBoost/Logit Boost/Gradient Boost/LGBM 차이
    1. AdaBoost (Adaptive)
        > - 훈련 과정 중 모델의 예측 능력을 향상 시킬 것으로 생각되는 특성들만 선택함으로써 차원수를 줄임과 동시에 필요 없는 특성들을 고려하지 않음으로써 잠재적으로 수행 시간을 개선함.
        > - `오분류 된 데이터`에 가중치를 부여
        > - 이전의 약한 classifier에서 잘못 분류된 `데이터`에 더 큰 가중치를 주어, 그 다음의 약한 classifier에 이를 반영하게끔 함. classifier 마다 연속적으로 이루어지는 이 과정을 'adaptive' 하다고 함.
        > - `데이터의 분포를 매번 바꿈`이 주요한 특징!
    2. Logit Boost
        
        > AdaBoost + Logistic Regression의 cost function을 추가적으로 더함.
    3. Gradient Boost
        
        > 이전 약한 classifier 에서 발생된 `잔차(gradient)`에 대해 다음의 약한 classifier를 적합 시킴.
    4. LightGBM
        > - 대부분의 트리 기반 알고리즘은 트리의 '깊이'를 효과적으로 줄이기 위해 균형 분할(level-wise) 사용. 즉, 최대한 균형 잡힌 트리를 유지하면서 분할하기 때문에 트리의 깊이 최소화(overfitting에 강함) 하는데 시간이 오래 걸리고 적은 데이터에서는 과적합 위험있음. `node(노드)` 중심 분할
        >- LGBM은 `leaf(리프)` 중심 분할 방식은 트리의 균형을 맞추지 않고, 최대 손실 값(max delta loss)을 가지는 리프 노드를 지속적으로 분할하면서 트리의 깊이가 깊어지고 비대칭적인 규칙 트리 생성함. 학습 반복할 수록 균형 트리 방식보다 손실 최소화 가능.
#
- [x] SVM 이란?
  - 확률을 직접 계산하지 않고, 데이터가 decision boundary (경계선) 를 넘는지 여부 검사함. 
  - 데이터와 경계선 사이의 최소 간격(margin)을 최대화 하는 경계선을 고른다. 왜냐하면 잘못 분류하는 것을 방지하기 위해.
  - 확률이 나오지는 않지만, 간격을 정하기 위해 경계선 근처의 점들만 고려하므로 모델이 가볍고 노이즈에 강하다.
  - 분류/예측 문제에 동시에 사용할 수 있고, 신경망 기법에 비해 과적합 정도가 덜함.
- [x] Margin 을 최대화 하면 어떤 장점?
    > 잘못 분류 하는 것을 방지할 수 있음. == 오류 최소화
#
- [x] PCA 란?
  > 여러 변수 간에 존재하는 상관 관계를 이용해 이를 대표하는 주성분(Principal Component)을 추출해서 차원을 축소함. 기존 데이터 정보 유실 최소화 하기 위해서 `가장 높은 분산을 가지는 데이터 축(PCA 주성분)(즉, 입력 데이터의 변동성이 가장 큰 축)`을 찾아서 이 축으로 차원을 축소함.
  - 분산이 데이터의 특성을 가장 잘 나타내는 것으로 간주
  - 단계
    1. 가장 큰 분산 기반으로 첫 번째 벡터 축 생성
    2. 두 번째 벡터 축은 1에 직교 벡터를 축으로.
    3. 다시 두 번째 축과 직교 벡터 축 생성
    4. ... 이렇게 생성된 벡터 축에 원본 데이터를 투영해서 벡터 축의 개수 만큼의 차원으로 축소한다.
   - `차원 축소`: 원본 보다 적은 차원이라서(새로운 차원의 데이터 생성) overfitting이 줄어든다. 차원이 증가할수록 데이터 포인트 간의 거리가 기하급수적으로 멀어지고 희소(sparse) 구조를 갖게 됨.
- [x] LDA 란? (Linear Discriminant Analysis)
  - 클래스 간 분산은 최대한 크게, 클래스 내부 분산은 최대한 작게 유지 하면서 차원 축소
  > PCA와 유사. 지도학습의 classification 에서 사용하기 쉽도록 `개별 클래스를 분별할 수 있는 기준을 최대한 유지하면서` 차원 축소
- [x] SVD (Singular Value Decomposition) 란? (선형 대수학 섹션에서 다시 설명)
  - PCA는 정방 행렬(dense matrix)만 가능, SVD는 행/열 크기가 다른 희소 행렬(sparse matrix)에 대한 차원 축소
- [x] Clustering 종류
  - [x] K-means (거리 기반, 실루엣 계수 사용)
    1. 군집 중심점(centroid)을 선택 후, 해당 중심에 가장 가까운 포인트들을 선택
    2. `centroid를 선택된 포인트의 평균 지점으로 이동`
    3. 이동된 중심점에서 다시 가까운 포인트를 선택...선택된 포인트의 평균 지점으로 centroid 이동...반복
    4. 모든 데이터 포인트에서 더 이상 centroid의 이동이 없을 경우에 반복을 멈추고 해당 중심점에 속하는 데이터 포인트들을 군집화
     - (단점) Feature 개수가 많을 경우에 군집 정확도 저하. 반복 횟수에 의존적임. 몇 개의 centroid 선택해야 하는지 가이드 필요.
   - [x] Mean-shift (밀도 기반, centroid 지정 필요 없음)
     - `Centroid를 데이터가 모여 있는 밀도가 가장 높은 곳으로 이동`시킴.
     - KDE(Kernel Density Estimation) 사용해서 PDF가 피크인 점을 centroid로 선정함. (확통 섹션 참고)
   - [x] GMM (Gaussian Mixture Model, 데이터를 여러 개의 가우시안 분포가 섞인 것으로 간주함.)
     - 데이터 세트에 대해서 이를 구성하는 여러 개의 정규 분포 곡선을 추출하고, 개별 데이터가 이 중 어떤 정규 분포에 속하는지 결정(확률 구함)
       - (1) 개별 정규 분포의 평균/분산 구하고,
       - (2) 각 데이터가 어떤 정규 분포에 해당되는지 확률 구함
     - `모수 추정` 이라고도 함 (확통 섹션 참고)
   - [x] DBSCAN (Density Based Spatial Clustering of Applications with Noise, 특정 공간 내에서의 데이터 밀도 차이 기반 군집화)
     - epsilon 주변 영역 내에 포함되는 최소 데이터 개수의 충족 여부에 따라 `데이터 포인트를 핵심/이웃/경계/잡음 포인트로 구분`하고, `특정 핵심 포인트에서 직접 접근이 가능한 다른 핵심 포인트를 서로 연결하며 군집화` 구성.
- [x] U-net 구조 및 특징
  - Image Segmentation 목적으로 제안된 `end-to-end learning 방식`의 `Fully-Convolutional Network 기반 모델`
    - `end-to-end learning`: 어떤 문제를 해결할 때 필요한 여러 스텝을 하나의 신경망을 통해 '재배치'하는 과정. 데이터 크기가 클 때 효율적. 즉, `데이터가 클 때, 두 단계로 나눠서 각각 네트워크를 구축/학습한 후 결과를 합침`.
  - 이미지의 전반적인 `컨텍스트 정보`를 얻기 위한 네트워크 + 정확한 `localization`을 위한 네트워크
    - `컨텍스트 정보`: 이웃한 픽셀 간의 정보. 이미지의 일부를 보고 이미지의 문맥을 파악.
  - `Contracting path` + `Expanding path`
    - `Contracting path` (수축구간, `downsampling`): 입력 `이미지의 컨텍스트 포착` 목적. VGG-based FCNs
      - 주변 픽셀들을 참조하는 범위를 넓혀가면서 이미지로부터 컨텍스트 정보 추출. 패딩 없어서 feature map 크기 감소. 
      - `FCN`: VGG 에서 마지막 3개의 FC layers 모두 CNN layers로 변경함. 출력 feature map은 원본 이미지의 위치 정보를 내포할 수 있음. 그러나 semantic segmentation의 최종 목적인 픽셀 단위 예측과 비교 했을때, FCN의 출력 feature map은 너무 coarse(거친, 알맹이가 큰?)함. 따라서 coarse map을 원본 이미지 크기에 가까운 dense map 으로 변환해줄 필요가 있음. --> Expanding path 
    - `Expanding path` (확장구간, `upsampling`): 세밀한 `localization 목적`. Contracting path 의 최종 feature map 으로부터 높은 해상도의 segmentation 결과를 얻기 위해 (높은 차원의 채널을 갖는) feature map을 Up-sampling을 여러번 하고(Coarse map -> dense prediction 위한 구조), contracting path에서 포착한 feature map의 context 와 결합(`Skip architecture`)함. 
- [x] 트리 기반 모델의 Feature importance 어떻게 구함?
  - Decision tree의 알고리즘: `CART(Classification and Regression Tree)`
    > 각 노드가 2개의 child node를 가지는 `binary tree를 불순도(impurity) 지표를 기준`으로 생성해 나가는 알고리즘.
      - 목적이 `분류`(classification)일 때에는 `불순도 지표로 Gini index(지니 계수) 및 Entropy`를 이용
      - 목적이 `회귀`(Regression)일 때는 `MSE 등을 이용해서 분산을 감소시키는 방향`으로 노드를 분할함
      - `불순도를 가장 크게 감소시키는 변수의 중요도가 가장 크게 됨`. CART 알고리즘은 이런 불순도를 이용해서 가장 중요한 변수들을 찾아냄.
    - > 야, 다 필요 없고, Decision tree는 CART 알고리즘을 따르며, 이는 binary split임. split의 기준은 불순도(지니 계수 - 섞여 있거나/분산이 높거나/다양한 클래스 구성이면 큼)가 작은(낮아지는) 방향으로 분할됨. 
  - `Gini Importance` (Gini impurity)
    - Scikit-learn 에서는 Gini Importance를 이용해서 각 feature의 중요도를 측정함.
    - `해당 노드에서 샘플들이 이질적으로 구성`되어 있을수록(모든 class에 골고루 분포되어 있을수록 or `분산이 클수록`) `지니 불순도(Gini Impurity)는 높아`짐. `한 클래스에 몰빵된 경우(순도가 높은 경우)에 Gini impurity가 낮아`짐. `불순도를 감소시키는 방향으로 노드를 생성하고 분류를 진행`함.
      - $G(N_j) = \sum_i^K p_i(1 - p_i) = 1 - \sum_{i=1}^K (p_i)^2$
      - K = class 총 개수
      - $p_i$ = 각 샘플이 해당 클래스에 속할 확률
      - $G(N_j)$ = 특정 노드 $N_j$에서 지니 불순도
  - 노드 중요도 `Node Importance` == `Information Gain`
    - 부모 노드의 가중치 불순도에서 자식 노드들의 가중치 불순도 합을 뺀 것.
      - $I(C_j) = w_j * G(C_j) - w_{j,left} * G(C_{j,left}) - w_{j,right}*G(C_{j,right})$
      - $I(C_j)$ = 노드 C<sub>j</sub>의 importance
      - $w_j$ = 전체 샘플 수에 대한 노드 C<sub>j</sub>에 해당하는 샘플 수의 비율, 즉 가중치
    - `Information Gain을 최대화 하는 feature를 기준으로 노드를 분할`해 나감.
    - 어떤 노드의 `Node Importance 값이 클수록`, 그 노드에서 `불순도가 특히 크게 감소`함을 의미.
  - `Feature Importance`
    -  (i번째 feature의 중요도)/(모든 feature들의 중요도 합)
       -  (i 번째 feature에 의해 생성된 노드들의 중요도를 합한 것(i번째 feature의 중요도) / 전체 노드의 중요도를 합한 것) / (모든 feature들의 중요도의 합).
- [x] Cut-mix 가 잘되는 이유?
  - Cutout은 regional dropout으로써, 학습이미지에서 유용한 픽셀들을 없애는 단점
  - Cutmix는 모델이 `객체의 차이를 식별할 수 있는 부분에 집중하지 않고, 덜 구별되는 부분(덜 중요한 부분) 및 이미지의 전체적인 구역을 보고 학습`하도록 해서 `일반화(generalization - 정상 데이터 들어왔을땐 더 좋겠지?)와 localization 성능을 높임`. 또한 train set과 test set의 분포가 다른 경우(out-of-distribution, OOD)와 이미지가 가려진 sample, adversarial sample에서의 robustness도 좋은 성능을 보임. 추가된 패치는 모델이 이미지의 부분만 보고도 오브젝트를 확인할 수 있는 localization ability 능력 향상
    - 아이디어 1. img1 과 img2를 random sampling 함
    - 아이디어 2. img2 에서 random fetch (전체에서 임의의 영역을 자른 부분 영역)를 copy해서 img1의 같은 위치에 붙여줌.
#
## 수행 했던 프로젝트들 중 기술 스택
#
- [x] EfficientNet의 우월함?
  - 적은 파라미터(작은 연산량) + 성능은 압도
  - 기본 블럭 및 뼈대는 MBConv + SENet(Squeeze-and-Excitation) optimization
  - 네트워크의 `Depth(#.layers)`, `Width(channel)(#.filters)`, `Resolution(image size)` 간의 balance를 통해 효과적이며 좋은 성능을 얻음: 복합 계수 제안(Uniformly scales all dimensions of depth/width/resolution) --> 궁극적인 목표는 모델의 compound `scale-up`
    - `Resolution` up: 세밀한(fine-grained) feature를 capture 하기 위해 사용 --> 정확도 증가
      - 왜? 입력의 해상도(resolution)가 작다면, 그렇지 않아도 입력의 해상도가 CNN을 거치면서(convolution & pooling) 특징이 소실되거나, 추출되지 않는 경우가 발생하는데 더 심해짐
      - `더 큰 receptive field` 필요 => layer 증가 필요
        - 왜? resolution 커질수록 유사한 픽셀 영역이 증가하기 때문
      - `더 큰 fine-grained pattern capture` 필요 => channel 증가 필요
        - 왜? fine-grained pattern 과 channel 의 크기(width) 연관
- [x] R-CNN vs. Fast R-CNN vs. Faster R-CNN
    1. `R-CNN` (Regions with CNN features)
         - ![스크린샷 2021-08-17 오후 3 35 26](https://user-images.githubusercontent.com/58493928/129809890-bfa9407b-8e7f-474c-ac26-e366579fd2c4.png)
       1. 바운딩 박스가 위치할만한 곳을 proposals: `이미지로부터 오브젝트가 존재할만한 위치에 바운딩 박스 proposal` (`selective-search`: `이미지에 대해서 2000개 정도의 각각 다른 region을 생성`한 뒤 물체가 있을 확률이 가장 높은 것을 뽑는 과정.)
            - 약 2000 개의 proposal region 생성
       2. 모든 proposals을 crop & resize(동일한 크기로 만듦. CNN에 넣기 위해)
       3. CNN에 넣음
       4. classifier(SVM - 분류)/바운딩 박스 regressor(바운딩 박스 교정) 처리
        - `단점`: 모든 proposal에 대해 CNN 거쳐야 하므로(`병목현상, 1CNN per region = 2000 CNN`) 연산량이 매우 많음.
    2. `Fast R-CNN`
         - ![스크린샷 2021-08-17 오후 3 35 40](https://user-images.githubusercontent.com/58493928/129809939-c97d6d16-75be-455d-ba30-bb13251a7f83.png)
       1. selective-search 같음
       2. 각 proposals이 CNN을 거치는것이 아니라, `전체 이미지에 대해 CNN을 한 번 거친 후 출력된 feature map 에서 객체 탐지 수행`
          1. 바운딩 박스를 한 이미지에서 2000개 추출하면 겹치는 부분이 엄청 많음. 이런 비효율을 해결하기 위해서 `전체 이미지`와 `RoI들`을 `함께 CNN` 통과 시킴
          2. 그 후 RoI pooling layer와 FC layers를 거쳐서  `RoI feature vector`를 뽑아냄.
          3. 각 RoI feature vector 마다 2개의 출력을 가짐.
          4. multi-task loss(classification loss + regression loss)를 이용해서 한 번에 학습이 이루어짐.
             1. 출력 1: classification (FC + Softmax)
             2. 출력 2: bounding box regressor
        - `단점`: Region proposal을 외부 알고리즘인 selective-search 로 수행하기 때문에 여전히 병목 현상 여전함.
    3. `Faster R-CNN`
        - ![스크린샷 2021-08-17 오후 3 35 55](https://user-images.githubusercontent.com/58493928/129809969-000d7092-fe44-4e45-b205-c9b2876d8dbb.png)
        - Region proposal을 selective-search가 아니라 CNN 기반의 `Region Proposal Networks(RPN)`을 사용함.
       1. 전체 이미지에 대해 CNN을 한 번 거친 후 출력된 feature map을 (1) RPN으로 넣음. (2) RoI pooling 에 전달
             1. RPN에서는 
                1. CNN을 통과한 feature map에 대해서, 슬라이딩 윈도우를 이용해 9개의 anchor box(3개의 서로 다른 크기, 3개의 서로 다른 ratio) 마다 가능한 바운딩 박스의 좌표(object proposals - 4k coordinates)와 그 점수(object score - 2k scores)
                2. RPN 에서는 object 인지 background 인지 분류만 함.
                   1. cls layer를 통과 시켜 `2k score` 출력: `바운딩 박스에 물체가 있는지/없는지` 2개 클래스 확률 분포 나타냄.
                   2. reg layer를 통과 시켜 `4k coordinates` 출력: `각 바운딩 박스를 어떻게 교정할지` 나타내는 벡터로 구성된 텐서.
       2. RPN을 거쳐 나온 것과 전체 이미지에 대해 CNN 거쳐 나온 feature map 에 대해서 RoI pooling 연산, classification/bbox position 뽑아냄.
     - ![스크린샷 2021-08-17 오후 3 36 30](https://user-images.githubusercontent.com/58493928/129809994-0da01617-8fc7-44ef-8239-c18e8988428d.png)
- [x] mAP in Image detection
  - mAP: mean Average Precision
  - `Precision`: 검출된 애들 중 정답
  - `Recall`: 검출 됐어야 하는 애들 중 정답
  - `Precision-Recall(PR) 곡선`: confidence level에 대한 threshold 값의 변화에 따라서 precision과 recall 값이 달라짐.
    - threshold 값 보다 낮은 confidence level 검출 결과 값은 무시.
    - x축: recall
    - y출: precision
      - recall 값의 변화에 따른 precision 값.
    - 어떤 알고리즘의 전반적인 성능 파악하기에는 괜찮지만, 서로 다른 두 알고리즘 성능을 정량적으로 비교하기에는 불편.
      - 그래서 `Average Precision` 등장. PR 그래프의 선 아래쪽 면적. 높을수록 성능이 전체적으로 우수. 보통 빨간색 처럼 단조적으로 감소하는 그래프로 변경후 면적 계산.
      - ![스크린샷 2021-08-17 오후 4 00 02](https://user-images.githubusercontent.com/58493928/129811613-da8015af-8e24-4c35-819e-3a959a52de43.png)
  - `mAP`: 물체 클래스가 여러 개인 경우, 각 클래스당 AP를 모두 합한 뒤 클래스의 개수로 나눔

