# ML Interview preparation (verbal)

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
## Linear Algebra
#
1. Linearly Independent 란?
2. Basis 와 Dimension 이란?
3. Null space 란 무엇?
4. Symmetric matrix 란?
5. Positive-definite 란?
6. Rank 란?
7. Determinant가 의미하는 바?
8. Eigen vector는 무엇?
9. Eigen vector는 왜 중요?
10. Eigen value 란?
11. SVD란 무엇? 중요한 이유는?
12. Jacobian matrix 란 무엇?
#
## Statistics/Probability (확률/통계) 및 Statistical Learning
#
1. Central Limit Theorem 이란?
2. Central Limit Theorem 은 어디에 쓸 수 있는지?
3. 큰 수의 법칙 이란?
4. 확률이랑 통계 다른 점
5. Marginal Distribution 이란 무엇인가?
6. Conditional Distribution 이란 무엇인가
7. Bayes rule/theory 란 무엇인가?
#
- [x] Bias 란 무엇인가?
   > 예측 값(추정된 파라미터들의 `평균`)과 실제 값(실제 파라미터)의 차이. 
   >> 상식적으로 '예측 값 - 실제 값' 이라고 생각하자.
- [x] Biased/Unbiased estimation의 차이는?
   - bias 는 `평균` 값의 차이를 다루기 때문에 같은 bias를 가진다고 하더라도 추정된 값들 끼리 차이가 클수도/작을수도 있다.
   > - Unbiased estimation: 파라미터 추정 평균에 대해서 bias가 0 인 경우
        >>> (ex) Gaussian distribution 에서 반복적으로 x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub> 값들을 뽑아서, 첫 번째 원소로 Gaussian의 평균을 추정하는 것
   > - Biased estimation: 파라미터 추정 평균의 bias가 0이 아닌 경우    
1. Bias, Variance, MSE란? 그리고 그들의 관계는?
2. Sample Variance 란 무엇인가?
3. Variance를 구할 때, N 대신에 N-1로 나눠주는 이유는?
4. Gaussian Distribution에서 MLE와 Sample Variance 중에 어떤 걸 사용해야 하는가?
5. Unbiased estimation은 무조건 좋은가?
6. Unbiased estimation의 장점은 무엇?
#
1. Binomial, Bernoulli, Multinomial, Multinoulli 란 무엇인가?
2. Beta Distribution, Dirichlet Distribution 이란 무엇인가?
3. Gamma Distribution은 어디에 쓰이는가?
4. Poisson Distribution은 어디에 쓰이는가?
5. Conjugate Prior 란?
- [x] Bias and Variance Trade-off 란 무엇인가?
  > 추정된 파라미터 값들의 차이가 클수록 값들이 멀리 퍼져있지만(큰 variance) 평균은 실제 파라미터와 비슷해짐(작은 bias)
  >> $$ MSE = E[({\hat \theta - \theta})^2 ]  = Bias + Variance $$
    >>> $$ Bias((E[\hat \theta] - \theta)^2) $$
    >>> $$ Variance = E[(E[\hat \theta] - \hat \theta)^2]$$
    - MSE가 동일하다면, Bias 값에 따라서 Variance 도 달라지게 되고 그 반대도 성립함.
    - 머신러닝에서 학습을 계속하면, 추정되는 파라미터는 실제 파라미터와 차이가 줄어들게 됨. 하지만 파라미터에 대해서 Bias 가 낮은것 보다는 MSE가 작은 경우가 제일 좋은 것임. 
    - Variance 를 높이려면 모델의 complexity를 높이면 됨. 

#
1. Confidence interval 이란?
2. Covariance/correlation 이란?
3. Total variation 이란?
4. Explained variation 이란?
5. Unexplained variation 이란?
6. Coefficient of determination 이란? (결정 계수)
7. Total variation distance 란?
8. P-value 란?
9. Likelihood-ratio test 란?
10. KDE란 무엇인가?
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
- [x] Receiver Operating Characteristic Curve 란?
  - 이진 분류 모델의 예측 성능 판단
  - FPR(x-축)이 변할 때, TPR(y-축, Recall)이 어떻게 변하는가.
    - $FPR = \frac {FP}{FP+TN}$
    - $TPR = \frac {TP}{TP+FN}$
  - AUC(Area Under Curve): ROC 곡선 밑의 면적. 1에 가까울 수록 좋다. FPR 작은 상태에서 큰 TPR 얻어야함.
- [x] Precision, Recall, Type I/II error
  - `Precision`: 양성으로 예측한 데이터 중 얼마나 정답? $precision = \frac {TP}{FP+TP}$
    - Type I error: FP. 양성으로 예측했지만 실제로는 음성. 
    - FP가 중요한 경우: 스팸 메일 분류. 실제로는 일반 메일인데 스팸 양성으로 예측해서 처리.
  - `Recall`: 실제 양성인 데이터 중에서 얼마나 정답? $Recall = \frac {TP}{FN+TP}$
    - Type II error: FN. 음성으로 분류했지만 실제는 양성.
    - FN이 중요한 경우: 암 진단/금융 사기. 실제로는 양성인데 음성으로 처리.
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
  - `Cross entropy`: 두 개의 확률 분포 p와 q에 대해 하나의 사건 x가 갖는 정보량(서로 다른 두 확률 분포에 대해 같은 사건이 가지는 정보량). `실제 확률 분포가 p일 때, 그와는 다른 어떠한 확률 분포 q로 부터 얻을 수 있는 정보의 양.`
    - $H(p,q) = -\sum_x p(x) * log{q(x)}$
    > ML에서 cross-entropy를 최소화 하는 문제는 결국, `데이터가 보여주는 분포와 모델이 학습한 분포의 각각의 확률이 같아지도록 하는 최적화`
  - `KL divergence`: 두 확률 분포 간의 정보량 차이를 나타냄. Cross-entropy는 Entropy 보다 항상 크고, p = q 일 때에만 같으므로 cross-entropy로 부터 entropy를 뺀 값을 두 분포 사이의 거리로 사용(). 사실 거리 함수는 아닌데(왜냐면 $D_{KL}(p||q)!=D_{KL}(q||p)$ 니까) 두 분포가 다르면 다를수록 큰 값을 가지며, 둘이 일치할 때에만 0을 갖기 때문에 거리와 비슷한 용도로 사용. 
    - 결국 cross-entropy minimization 문제는 KL divergence를 최소화 하는 문제와 동치임.
      - H(p)는 p의 엔트로피, 즉 우리가 가지고 있는 데이터의 분포이며 학습과정에서 바뀌지 않음. 따라서 q에 대해서 cross-entropy를 최소화 한다는 것은 KL divergence를 최소화 한다는 의미. 따라서 p를 근사하는 q의 확률분포가 최대한 p와 같아질 수 있도록 모델의 파라미터를 조정하는 것.
    - $D_{KL}(p||q) = H(p,q) - H(p) = \sum_x p(x)log{\frac{1}{q(x)}} - p(x)log{\frac{1}{p(x)}} = \sum_x p(x)log{\frac{p(x)}{q(x)}}$
- [x] Mutual Information 이란?
  - joint probability p(x,y) 와 두 marginal 의 곱(p(x)q(x)) 사이의 KL divergence
  - $MI(X;Y) = D_{KL} (p(x,y)||p(x)q(y))$
  - 두 변수 `X와 Y가 독립`이면 $p(x,y) = p(x)q(y)$ 이므로 두 `분포 사이의 거리가 0 이면(KL divergence가 0이 됨)` 독립일 것이고 `mutual information이 없다`고 볼 수 있다. 반면, 두 분포 사이의 거리가 멀다는 것은(0보다 큰 어떤 값) 결과적으로 두 변수 X와 Y 사이에 공유하는 정보가 많아서 독립이 아니고 mutual information이 발생.
- [x] Cross-Entropy loss 란?
  - `negative` maximum `log` likelihood estimation
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
  - `Generative model`: 각 class의 분포에 주목. Gaussian Mixture Model, Naive Bayes, Restricted Boltzmann Machine. GMM을 예로 들면(clustering 방법 중 하나) 모든 class 들이 Gaussian distribution의 mixture 형태로 주어진다고 가정함. 이 경우는 `데이터와 class의 Joint probability를 계산함`.
  - `Discriminative model`: 두 class가 주어진 경우, 이들 class의 차이점에 주목. Logistic regression, LDA, SVM, Boosting, 신경망 등.
#
- [x] 분류와 회귀의 차이
  - 분류(classification): 예측값(결과값)이 `이산형` (ex. 카테고리 같은 것)
  - 회귀(regression): 예측값(결과값)이 `연속형`(ex. 숫자 같은 것)
  - Tree 기반의 회귀
    - 분류: `leaf node`에서 예측 결정 값을 만드는데, 특정 `class label 결정`
    - 회귀: `leaf node에 속한 데이터 값의 평균 값`을 구해 `회귀 예측 값을 계산`
- [x] 회귀란? (Regression)
  - 여러 개의 독립 변수(x1, x2..)와 한 개의 종속 변수(y) 간의 상관 관계 모델링
    - 결국 최적의 회귀 계수(w)를 찾는 것.
  - $y = w_1 * x1 + w_2 * x2 + ... + w_n * x_n$ 
  - 회귀 계수의 선형/비선형 여부에 따라 '선형 회귀, 비선형 회귀'가 결정됨. (독립 변수 x, 종속 변수 y와는 상관 없음.)
    - 독립 변수 x의 갯수 (1개 - 단일 회귀, 여러개 - 다중/다항 회귀 polynomial)
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
    > - Underfitting: 학습 데이터에 대해서 충분히 학습하지 못한 경우. 학습 데이터에 대한 성능이 테스트 데이터에 대한 성능 보다 좋지 못한 경우. D < M
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
          - 0 미만의 값은 다음 레이어에 전달하지 않음. 
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
    - `padding`: 합성곱 연산 수행시 출력 feature map이 입력 feature map 보다 계속 작아지는 것을 막음. 모서리 feature 잡아냄.
      - `same`: 아웃풋이 인풋과 동일한 길이를 갖도록 인풋 패딩
      - `valid`: 패딩 없음
      - `causal`: 입력의 왼쪽에 0을 알맞게 패딩하고 valid 처리. 시계열 모델에서 미래 시퀀스 보지 않기 위함.
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
        - `원인`: tanh 가 기본 활성화 함수인 이유(= 수렴하는 활성화 함수 사용하는 이유): 동일한 가중치가 모든 타임스텝에서 사용되기 때문. 그래서 증가-->계속 증가 // 감소 --> 계속 감소
        - 해결책1: `LayerNormalization`
          - 특성 차원에 대해 정규화
          - 입력마다 하나의 스케일과 이동 파라미터를 학습(BN과 비슷)
          - 샘플에 독립적으로 타임스텝마다 동적으로 필요한 통계를 계산할 수 있음. 즉, 훈련과 테스트에서 동일한 방식으로 작동함 (BN과 다름)
        - 해결책2: `dropout` or `recurrent_dropout` 매개변수 사용
          - dropout: (타임스텝마다) 입력에 적용하는 드롭아웃 비율 정의
          - recurrent_dropout: (타임스텝마다) 은닉 상태에 대한 트롭아웃 비율 정의
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

- [x] [Newton's method](https://darkpgmr.tistory.com/58?category=761008) 이란 무엇?
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
  - 오차(loss)를 개별 가중치(w 변수)에 대해 편미분 해서 구해지는 기울기(gradient)와 학습률(learning rate, eta) 활용.
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
  - 1. Feed forward(순전파) 수행: 데이터를 입력 받은 모델은 (초기화된) 가중치를 이용해 예측값을 출력
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
  - 이처럼, 학습/테스트 데이터셋 간의 차이에 대한 문제(Covariate Shift)를 각 mini batch간 input 데이터의 차이에 의한 문제로 확장 시킨 것을 `Internal Covariate Shift` 라고 함.
    - 신경망에서는 보통 모든 학습 데이터를 한 번에 사용하지 않고 mini batch를 사용하는데, 각 step에서 사용되는 학습 데이터는 매번 달라지게 됨. 이렇게 배치 간의 분포가 다른 경우를 `internal covariate shift` 라고 함.
    - 신경망 각 층을 통과할 때마다 입력 데이터의 분포가 조금씩 변경/누적되는 현상 발생. 
      - 왜냐하면, 각 Layer 마다 input을 받아서 linear combination 을 구한 후 activation function을 적용해서 output을 구하는 작업이 이루어지기 때문. 결과적으로 이 때문에 각 layer의 input data 분포가 달라지게 되며, 뒷단에 위치한 layer 일 수록 변형이 누적되어 input data 의 분포가 상당히 많이 달라지게 됨.
      - 따라서 모델들의 parameter 들이 일관적인 학습을 하기가 어려워짐.
  - ![스크린샷 2021-08-13 오전 12 01 55](https://user-images.githubusercontent.com/58493928/129317781-5cc57360-9d30-4d1e-a951-4dadbede89ab.png)
  - 게다가 weight가 특이하게 큰 것이 있으면 영향력이 커짐.
    - 그래서 각 층을 거칠 때마다 z-score로 다시 scaling 함 ==> `BatchNormalization`
- [x] Batch Normalization 은 무엇이고 왜 하는지?
  - `Regularization` 효과
  - `Internal Covariate Shift` 해소 하기 위해.
  - 각 층을 거칠 때마다 z-score로 다시 scaling 함. 
    - weight 자체가 작아지고 변동성이 작기 떄문에 GD가 빠르게 수렴할 수 있음.
    - Batch 마다 하며, scaling 파라미터와 shift 파라미터는 학습으로 동적 갱신함.
    - 테스트 데이터에 대한 BN은 학습 때와 다름.
      - 해당 mini-batch에 대한 표본 평균과 표본 분산을 쓰는 것이 아니라(각 mini-batch 마다 표본 평균과 표본 분산이 다르니까 normalization 결과가 달라지잖니..), 학습 과정의 K개의 mini-batch에서 얻은 K개의 표본 평균을 평균낸 값(learning mean)과 K개의 표본 분산을 평균낸 값(learning variance)을 사용함.


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
  - PCA는 정방 행렬(dense matrix)만 가능, SVD는 행/렬 크리가 다른 희소 행렬(sparse matrix)에 대한 차원 축소
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