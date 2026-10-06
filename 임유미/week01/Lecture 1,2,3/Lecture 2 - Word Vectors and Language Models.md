# Lecture 2 - Word Vectors and Language Models

# 2강. Word Vectors, Word Senses, and Neural Classifiers

## Gradient Descent

> Cost function J(θ)를 최소화하는 알고리즘
> 

현재 θ에서 gradient를 계산하고, **negative gradient 방향으로 작은 step**만큼 이동하는 과정을 반복한다.

```jsx
θ_new = θ_old − α∇J(θ)
```

α는 step size(learning rate)

너무 크면 발산하거나 최솟값을 지나치고, 너무 작으면 학습이 느리다.

## Stochastic Gradient Descent (SGD)

SGD는 **window(또는 mini-batch)를 샘플링해서 gradient를 계산하고 바로 update**한다. Gradient에 noise가 있지만 훨씬 빠르며, 신경망 학습에서는 mini-batch SGD가 표준이다.

## Skip-gram과 CBOW

- **Skip-gram (SG)**: 중심 단어가 주어지면 주변(context) 단어를 예측한다.
- **CBOW**: 주변 단어들로 중심 단어를 예측한다.

Loss로는 naive softmax(단순하지만 비쌈), hierarchical softmax, negative sampling이 있다.

## Negative Sampling

Softmax의 정규화 항은 vocabulary 전체를 합산하므로 비용이 크다. 대신 **실제 (center, outside) 쌍과 K개의 random noise 쌍을 구분하는 binary logistic regression**을 학습한다.

```jsx
J = −log σ(uₒᵀv_c) − Σ_k log σ(−u_kᵀv_c)
```

Negative sample은 unigram 분포 U(w)를 **3/4승**하여 뽑는다. 이렇게 하면 빈도가 낮은 단어가 조금 더 자주 선택된다. 한 window에서 갱신되는 단어는 소수이므로 gradient가 매우 sparse하고, 따라서 sparse update(해당 행만 갱신)가 중요하다.

## Co-occurrence Matrix

Corpus 전체에서 함께 등장하는 횟수를 직접 세는 방법이다.

- **Window 기반**: word2vec처럼 주변 단어를 세며 syntactic/semantic 정보를 포착한다.
- **Word-document 기반**: 일반적인 토픽을 포착하며 Latent Semantic Analysis로 이어진다.

단순 count 벡터는 vocabulary와 함께 차원이 커지고, sparse하며, 이를 쓰는 모델은 덜 robust하다. 그래서 25~1000차원의 **dense vector**로 줄이는 것이 목표다.

## SVD와 Count Scaling

Co-occurrence matrix X를 UΣVᵀ로 분해하고 **k개의 singular value만 유지**하면 least squares 관점에서 최적의 rank-k 근사가 된다. 큰 행렬에서는 계산 비용이 크다.

Raw count에 바로 SVD를 하면 잘 안 되며, the/he/has 같은 function word의 영향이 너무 크다. 해결책은 다음과 같다.

- 빈도에 log 적용, min(X, t) 상한 적용
- function word 제거
- 가까운 단어에 가중치를 더 주는 ramped window
- count 대신 Pearson correlation 사용

이렇게 scaling한 COALS 모델에서는 "행위자(doer)" 같은 의미 성분이 벡터 공간의 **선형 방향**으로 나타난다(drive→driver, teach→teacher).

## GloVe

**핵심 통찰: co-occurrence 확률의 비율이 의미 성분을 인코딩한다.** 예를 들어 P(x|ice)/P(x|steam)은 solid에서는 크고(8.9), gas에서는 작으며(0.085), water나 fashion처럼 무관한 단어에서는 1에 가깝다.

이 비율을 벡터 공간의 선형 성분으로 담기 위해 **log-bilinear 모델**을 쓴다.

```jsx
wᵢ · wⱼ = log P(i|j),  wₓ · (wₐ − w_b) = log P(x|a)/P(x|b)

Loss: J = Σ f(X_ij)(wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ − log X_ij)²
```

f는 고빈도 단어의 영향을 제한하는 함수이다. 학습이 빠르고 큰 corpus에도 확장 가능하다.

## Word Vector 평가

- **Intrinsic**: 특정 하위 과제로 평가한다. 계산이 빠르고 시스템을 이해하는 데 도움이 되지만 실제 과제와의 상관관계가 확인되어야 한다.
    - **Analogy** (man:woman :: king:?): x_b − x_a + x_c와 cosine similarity가 가장 높은 단어를 찾으며, 이때 입력 단어는 검색에서 제외한다.
    - **유사도 상관**: 사람의 판단(예: WordSim353)과 벡터 거리의 상관을 본다.
- **Extrinsic**: 실제 과제(예: NER)에 넣어 성능을 본다. 시간이 오래 걸리고 어느 부분이 문제인지 불분명하다. 하나의 하위 시스템만 교체해서 성능이 오르면 성공이다.

## Word Senses

대부분의 단어는 여러 의미를 가진다(예: pike). 한 벡터에 의미가 섞이는 문제를 두 방식으로 다룬다.

- **Multiple prototypes (Huang et al. 2012)**: 단어의 context window를 clustering하여 bank₁, bank₂처럼 sense별 벡터를 만든다.
- **Linear superposition (Arora et al. 2018)**: 표준 embedding의 벡터는 각 sense 벡터의 **가중합**(가중치는 sense 빈도 비율)이다. Sparse coding 아이디어로 흔한 sense는 분리해낼 수 있다.

## NER과 Window Classification

**NER**은 텍스트에서 이름(PER, LOC, ORG, DATE 등)을 찾아 분류하는 과제이다. 같은 단어(Paris)도 문맥에 따라 라벨이 달라진다.

**Window classification**은 중심 단어 주변 window의 word vector들을 **concatenate**(x ∈ ℝ^(5d) 등)하여 각 단어의 클래스를 분류한다.

## Neural Classifier

일반 softmax 분류기는 W만 학습하고 **선형 결정 경계**만 만들 수 있다. 신경망은 W와 함께 **단어 표현(distributed representation)도 학습**하며, 여러 층으로 데이터를 반복해 re-represent/compose하여 **비선형 분류기**를 만든다(마지막 층 직전 표현에 대해서는 선형).

## Cross Entropy Loss

```jsx
H(p, q) = −Σ p(c) log q(c).
```

정답 분포 p가 one-hot이면 남는 항은 정답 클래스의 **negative log probability**, 즉 −log p(yᵢ|xᵢ)이다. PyTorch에서는 `torch.nn.CrossEntropyLoss()`를 쓴다.

## 신경망의 구조

- 뉴런 하나는 **binary logistic regression 유닛**이다: h = f(wᵀx + b). w는 가중치, b는 bias, f는 비선형 활성화 함수이다.
- 신경망은 **여러 logistic regression을 동시에 실행**하는 것이다. 중간 hidden 변수가 무엇을 예측해야 하는지는 미리 정하지 않으며, **최종 loss가 이를 결정**한다.
- 층의 행렬 표기: `z = Wx + b, a = f(z) (f는 element-wise로 적용)`
- NER용 binary 신경망: `h = f(Wx + b), s = uᵀh, 확률 = σ(s)`

## 비선형성이 필요한 이유

비선형 함수가 없으면 여러 층이 W₁W₂x = Wx처럼 **하나의 선형 변환으로 합쳐져** 층을 쌓는 의미가 없다. 비선형 함수(sigmoid, tanh, ReLU 등)가 있어야 층을 쌓아 복잡한 함수를 근사할 수 있다.