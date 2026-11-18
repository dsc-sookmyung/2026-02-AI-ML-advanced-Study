# Lecture 1 - Intro and Word Vectors

# 1. 인간 언어와 단어의 의미

## 의미(meaning)와 Denotational Semantics

언어학에서 가장 흔한 의미 관점

기호(signifier) ⟺ 지시 대상(signified, 개념 또는 사물)의 대응

## WordNet

WordNet은 동의어 집합과 상위어를 정리한 단어장과 같음

**한계**

- 뉘앙스가 빠짐 (예: proficient는 good의 동의어지만 일부 문맥에서만 맞음)
- 새로운 단어 의미를 반영하지 못하고 최신 상태 유지가 불가능
- 사람이 직접 만들고 관리해야 하며 주관적임
- 단어 유사도를 정확히 계산할 수 없음

## One-hot 벡터 (Localist Representation)

단어를 이산적인 기호로 보고, 해당 단어 위치만 1, 나머지는 0인 벡터로 표현. 벡터 차원은 어휘 수와 동일함.

**한계**

- 벡터 자체에 유사도 개념이 없어 "Seattle motel" 검색이 "Seattle hotel" 문서와 연결되지 못함
- 해결 방향: 유사도를 벡터 자체에 학습시키기

## Distributional Semantics (분포 의미론)

**단어의 의미는 그 단어 근처에 자주 등장하는 단어들로 결정됨**

## Word Vector (Word Embedding)

각 단어에 대해 밀집(dense) 벡터를 만들고, 비슷한 문맥에 나오는 단어끼리 벡터가 비슷하도록 학습한다. 유사도는 벡터의 내적(dot product)으로 측정한다.

---

# 2. Word2Vec 소개

## Word2Vec 개요

> 단어 벡터를 학습하는 프레임워크
> 
1. 큰 텍스트 코퍼스를 준비한다.
2. 고정된 어휘의 모든 단어를 벡터로 표현한다.
3. 텍스트의 각 위치 t마다 중심 단어 c와 주변(outside/context) 단어 o를 정한다.
4. c와 o의 벡터 유사도로 P(o | c)를 계산한다.
5. 이 확률이 커지도록 벡터를 계속 조정한다.

## 목적 함수 (Objective Function)

likelihood는 모든 위치와 window 내 주변 단어의 확률 곱이다.

```
L(θ) = Π_t Π_{-m≤j≤m, j≠0} P(w_{t+j} | w_t ; θ)
```

이를 최대화하는 대신 평균 음의 로그 likelihood를 최소화한다. 음수를 붙이는 이유는 최소화 문제로 만들기 위함이다.

```
J(θ) = -(1/T) Σ_t Σ_{-m≤j≤m, j≠0} log P(w_{t+j} | w_t ; θ)
```

- 목적 함수 최소화 ⟺ 예측 정확도 최대화
- J(θ)는 cost 또는 loss function이라고도 부름

## 단어당 두 개의 벡터

P(o | c)를 계산하기 위해 단어마다 벡터를 두 개 사용한다.

- v_w: w가 중심 단어일 때의 벡터
- u_w: w가 주변(context) 단어일 때의 벡터

## Softmax 예측 함수

```
P(o | c) = exp(u_o^T v_c) / Σ_{w∈V} exp(u_w^T v_c)
```

1. 내적 u_o^T v_c로 o와 c의 유사도를 계산 (내적이 클수록 확률이 큼)
2. 지수함수로 모든 값을 양수로 변환
3. 전체 어휘에 대해 합으로 나누어 확률 분포로 정규화

softmax는 임의의 실수 값을 확률 분포로 바꾼다. "max"는 가장 큰 값의 확률을 증폭하기 때문이고, "soft"는 작은 값에도 일부 확률을 남기기 때문이다.

---

# 3. 목적 함수의 미분 (Gradient)

## 미분 대상

중심 단어 벡터 v_c에 대한 log P(o | c)의 기울기를 구한다. 벡터에 대한 미분이며, 한 번에 한 변수(성분)씩 계산하면 복잡할 때 도움이 된다.

## 로그 분해

log의 성질로 분자와 분모를 나눠서 미분한다.

```
∂/∂v_c log P(o|c) = ∂/∂v_c log exp(u_o^T v_c)  -  ∂/∂v_c log Σ_w exp(u_w^T v_c)
                          ①                                ②
```

## 항 ①

log와 exp가 서로 역함수이므로 ∂/∂v_c (u_o^T v_c) = u_o 이다.

## 항 ②와 Chain Rule

chain rule을 적용하고, 미분을 합 안으로 옮기며, 합의 인덱스를 x로 바꿔서 계산한다.

```
② = (1 / Σ_w exp(u_w^T v_c)) · Σ_x exp(u_x^T v_c) u_x
```

## 최종 결과: Observed − Expected

```
∂/∂v_c log P(o|c) = u_o - Σ_x P(x|c) u_x
```

- u_o: 실제로 관찰된 주변 단어 벡터 (observed)
- Σ_x P(x|c) u_x: 모델 확률로 가중평균한 주변 단어 벡터, 즉 기댓값 (expected)
- 현재 모델의 예측과 실제 관찰의 차이가 업데이트에 반영된다.
- 여기까지는 중심 벡터 v_c의 미분이며, 주변 벡터 u에 대한 미분도 비슷하게 구해서 모든 파라미터의 미분을 얻는다.

---

# 4. 최적화: 경사하강법

## Gradient Descent

비용 함수 J(θ)를 최소화하는 알고리즘이다. 현재 θ에서 기울기를 계산하고, 음의 기울기 방향으로 작은 걸음을 옮기는 과정을 반복한다.

```
θ_new = θ_old - α ∇_θ J(θ)
```

- α: step size 또는 learning rate
- 목적 함수가 convex가 아닐 수도 있지만 실제로는 잘 동작한다.

## Stochastic Gradient Descent (SGD)

J(θ)는 코퍼스의 모든 window에 대한 함수라서 ∇J(θ) 계산이 매우 비싸고, 업데이트 한 번을 위해 너무 오래 기다려야 한다. 이는 거의 모든 신경망에서 매우 나쁜 방식이다.

- 해결: window를 반복적으로 샘플링하고 매번 바로 업데이트 (Mini-batch Gradient Descent)