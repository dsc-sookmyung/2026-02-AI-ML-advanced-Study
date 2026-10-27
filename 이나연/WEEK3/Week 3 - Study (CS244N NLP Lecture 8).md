# Week 3 - Study (CS244N NLP Lecture 8)

### 📒 강의 개요

Lecture 8 - Transformer : 

https://www.youtube.com/watch?v=LWMzyfvuehA&list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D&index=8

## 8강. **Self-Attention and Transformers**

# Self-Attention과 Transformer 학습 노트

> **Self-attention과 Transformer는 NLP 발전의 중요한 foundation이다.**

---

## 1. RNN의 한계와 Self-Attention의 등장 배경

### 접근 방향

기존과 **같은 목표(same goals), 다른 building blocks**. 모델 전체를 바꾸기보다는, 모델을 구성하는 building blocks를 다시 고민하자는 것에서 시작되었습니다.

### RNN/LSTM의 한계: Linear Interaction Distance

| 문제 | 설명 |
|---|---|
| **먼 거리 의존성** | 멀리 떨어진 두 단어가 상호작용하려면 시퀀스 길이에 비례하는 수($O(\text{sequence length})$)만큼 RNN 스텝을 순차적으로 거쳐야 함 |
| **기울기 문제** | 여러 스텝을 거치며 gradient 손실/폭발이 발생해 long-distance dependencies 학습이 어려움 |
| **병렬화 불가** | 시점 순서에 종속되어 시퀀스 길이만큼의 연산을 병렬화할 수 없음 → GPU 활용이 어려움 |

### Self-Attention의 해결 방식

- 기존 어텐션은 encoder-decoder 구조에서 동작했지만, **Self-Attention은 단일 시퀀스(단일 문장) 내부**에서 동작합니다.
- 시퀀스 내 모든 단어가 거리와 무관하게 **한 번의 연산**으로 직접 상호작용합니다.
- 문장의 각 토큰 임베딩을 독립적으로 한 번에 곱하므로, 선행 의존이 없습니다.

> ⇒ **병렬 처리가 가능하고, 장거리 의존성 문제를 구조적으로 해결**

**Intuition**: 어텐션은 멀리 볼 수 있고, 시퀀스 index에 의존적이지 않다. (⇒ 병렬 계산 가능)

---

## 2. Query, Key, Value 기반 Self-Attention

### 개념

파이썬 딕셔너리의 Key-Value 조회를 **연속적인 벡터 공간으로 확장**한 개념입니다. 각 단어 임베딩을 학습 가능한 가중치 행렬로 Q, K, V 세 벡터로 변환합니다.

- 사전에는 key들이 있고, 각 key는 value와 대응됩니다. query를 key와 대조해 value를 찾습니다.
- **비유**: key = 메뉴판 / value = 음식 / query = 주문
- 단, self-attention은 weight를 활용해 **부드럽게(soft) 대조**합니다.

> 🍜 **Soft Lookup 비유**
> "짬뽕 70% + 탕수육 30% 느낌의 퓨전 음식을 만들어줘"라고 쿼리를 날리면, 주방(Value)에서 그 비율대로 재료를 섞어 **새로운 한 그릇(새로운 임베딩 벡터)** 을 만들어줍니다.

### 연산 과정

1. 단어 임베딩에 각각 가중치 행렬 $W^Q, W^K, W^V$를 곱해 $q, k, v$ 생성
2. 쿼리와 모든 키의 내적 $QK^T$로 **유사도 점수** 계산
3. **Softmax**로 정규화 (합이 1인 확률값)
4. 밸류 벡터들과 **가중합**하여 문맥이 반영된 새 표상 생성

$$\text{Attention}(Q, K, V) = \text{softmax}(QK^T)V$$

- 흐름: $q_i \cdot k_j$ (내적) ⇒ 점수 ⇒ softmax ⇒ $\alpha_{ij}$
- **$\alpha_{ij}$**: $i$번째 단어가 새로운 표상을 만들 때, **$j$번째 단어의 정보(Value)를 얼마나 가져올지** 결정하는 가중치
- 즉 $i$번째 단어가 문장 속 다른 $j$번째 단어들을 둘러보며 "누구한테 얼마나 집중해서 정보를 가져올까?"를 계산하는 것

> 💡 **질의응답**
> **Q. 자기 자신에 대해서도 연산하면 내적값이 크게 나와서 무조건 자기 자신만 보게 되지 않을까?**
> A. $Q$와 $K$를 만드는 가중치 행렬($W^Q, W^K$)이 서로 다르기 때문에, 자기 자신을 볼지 말지는 모델이 학습을 통해 유연하게 결정합니다. 즉 $Q$와 $K$를 분리해 파라미터화했기 때문에 '자기 자신을 얼마나 참조할지'조차 데이터로부터 스스로 학습할 수 있습니다.

---

## 3. Positional Encoding (위치 인코딩)

### 첫 번째 장벽: Sequence Order

- 예: *"Zuko made his uncle"* vs *"his uncle made Zuko"*
- 셀프 어텐션은 **집합(Set) 연산**처럼 동작해, 순서가 바뀌어도 동일한 출력을 냅니다. → 단어 순서 정보 반영 불가

### 해결: 위치 벡터를 직접 더하기

sequence index를 벡터($p_i$)로 만들어 입력 단어 임베딩에 **직접 더해줍니다.**

$$x_i \leftarrow x_i + p_i$$

- 매 레이어가 아니라 **처음 레이어에서만** 수행합니다.
- 모델이 단어의 절대적·상대적 위치를 구별할 수 있게 됩니다.

### 두 가지 방식 비교

| | **Sinusoidal** | **Learned Embedding ($p_i$)** |
|---|---|---|
| 방식 | sin/cos 주기 함수 기반 | 학습 가능한 파라미터 |
| 장점 | absolute position에 얽매이지 않음, 이론적으로 학습 때보다 긴 시퀀스 반영 가능 | **Flexibility** |
| 단점 | 수식 고정(not learnable), 실제로는 긴 시퀀스 확장 능력이 떨어져 보통 그대로 쓰지 않음 | 1~n까지만 가능 (시퀀스 길이 n 제한) |
| 실사용 | 드묾 | **대부분의 시스템에서 사용** |

---

## 4. Position-wise Feed-Forward Network (FFN)

### 두 번째 장벽: 비선형성 부재

- 셀프 어텐션 블록은 선형 변환과 가중 평균으로만 구성되어 있어 **비선형 표현력이 부족**합니다. (attention을 계속 수행해도 그냥 벡터 계산의 반복)

### 해결

- 어텐션을 통과한 **각 위치의 출력 벡터**를 독립적인 **2층 MLP**에 통과시켜 비선형성을 부여합니다.
- 병렬 연산이 가능해 **GPU 효율이 높습니다.**

> 📌 **역할 구분**
> - **Self-attention**: 모든 단어를 둘러보며 정보를 **섞음**
> - **FFN**: 각 단어가 **독립적으로** 자기 위치의 벡터 하나만 가지고 MLP를 통과

---

## 5. Masked Self-Attention

### 세 번째 장벽: 미래 정보 유출

- 다음 단어를 예측해야 하는데, self-attention은 전체 $n \times n$ 행렬을 보고 계산하므로 **정답(미래 토큰)을 미리 참조하는 '치팅'** 이 가능해집니다.

### 해결: Masking the future

- 각 timestep마다 과거 단어만 참조하도록 따로 계산하는 방식은 **비효율적**입니다.
- 대신 모든 쌍의 어텐션 스코어 행렬을 **병렬로 계산**한 뒤, 미래 위치의 점수를 **$-\infty$** 로 치환합니다.
- softmax를 거치면 해당 가중치가 **0**이 됩니다.

### 📋 지금까지의 정리

| 문제 | 해결책 |
|---|---|
| 단어 순서 정보 부재 (set 연산처럼 동작) | **Positional Encoding** 추가 |
| 비선형성 부족 (선형 변환 + 가중 평균뿐) | **Position-wise FFN** 적용 |
| 미래 시점 정보 유출 | **Masking** (attention 가중치를 0으로) |

> 실제로는 **self-attention과 FFN을 반복**합니다.

---

## 6. Multi-Head Attention

> Transformer decoder가 LM의 구성 시스템이지만 endpoint(정답)는 아니며, 기존 구조를 어떻게 개선해 나갔는지 위주로 보기.

### 행렬 연산으로 보기 (GPU 효율)

$X \in \mathbb{R}^{n \times d}$ 일 때,

1. $(XW^Q)(XW^K)^T$ → 모든 self-attention 점수 ($n \times n$)
2. softmax 후 $V$와 곱 → 출력 ($n \times d$)

이렇게 행렬 형태로 GPU에서 한 번에 수행할 수 있습니다.

### 왜 Multi-Head인가?

한 단어가 문맥을 파악할 때는 **구문 구조, 수식 관계, 특정 개체명** 등 여러 관점에서 정보를 동시에 참조해야 합니다. 단일 어텐션 가중합만으로는 한계가 있습니다.

⇒ 여러 개의 Q, K, V 행렬에 대해 **multiple attention "heads"** 를 정의합니다.

- 모델 차원 $d$를 헤드 수 $h$로 나눈 **저차원 공간($d/h$)** 으로 투영
- 각 헤드가 서로 다른 문맥적 특징을 **독립적으로 병렬 학습**
- 결과를 **Concatenate** 한 뒤 최종 선형 변환으로 결합
- 각 head는 서로 다른 관점·이유로 "look"하며 value를 다르게 구축 (각 block은 share되지 않고 independent)

> 💡 **파라미터 수는 변하지 않는다**
> - 헤드 1개당 $W_i^Q \in \mathbb{R}^{d \times (d/h)}$
> - 헤드 $h$개 총합: $h \times (d \times \frac{d}{h}) = d \times d$
> - 헤드가 1개일 때($W^Q \in \mathbb{R}^{d \times d}$)와 **파라미터 총량이 동일**
> - 예: $d=512$, $h=8$ → 각 헤드 차원 $d/h = 64$ (512차원 벡터가 64차원으로 매핑)

---

## 7. Scaled Dot-Product Attention

### 문제

- 임베딩 차원 $d_k$가 커질수록 무작위 벡터 간 **내적 값의 크기와 분산이 커집니다.**
- 그 결과 softmax 입력이 극단적으로 커져 **기울기 소실(Vanishing Gradient)** 이 발생합니다.

### 해결

내적 점수를 $\sqrt{d_k}$로 나눠 softmax 입력의 **분산을 일정하게 유지**합니다.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

⇒ 초기 학습 시 어텐션 가중치가 지나치게 치우치지 않고 원활하게 수렴합니다.

---

## 8. Add & Norm: Residual Connection과 Layer Normalization

> Optimization tricks를 사용하자.

### Residual Connection (잔차 연결)

- 서브레이어의 입력 $x$를 출력에 **직접 더해줍니다.** 레이어는 입출력 간의 **'잔차'만 학습**하면 됩니다.
- 기울기가 1로 역전파되어, 깊은 층에서도 기울기 소실 없이 안정적으로 학습됩니다.
- 레이어를 통과하며 vanishing이 생겨도, 더해준 값이 있으므로 손실 없이 전달됩니다.

### Layer Normalization (레이어 정규화)

- 배치나 토큰 간 통계를 공유하지 않고, **각 단어 벡터 내부 차원들의 평균과 표준편차**로 독립적으로 정규화합니다.
- 평균 0, 분산 1 (Unit Mean, Variance)
- 학습 가능한 추가 파라미터로 조절: **gain($\gamma$), bias($\beta$)**
- 층 간 불필요한 활성화 값 변동을 억제하고 **수렴 속도를 높입니다.**

---

## Encoder-Decoder 모델

- **self-attention과 Add&Norm을 반복**합니다.
- **Encoder는 masking이 없습니다.** 문맥을 최대한 파악하기 위해 단어들을 **양방향(bidirectional)** 으로 전부 참조해야 하기 때문입니다.

---

## 9. Cross-Attention (교차 어텐션)

인코더-디코더 구조(예: 기계 번역)에서 디코더가 **입력 원문(Source) 문장의 정보**를 가져오기 위해 쓰는 핵심 메커니즘입니다.

| 구분 | 출처 |
|---|---|
| **Query** | 디코더의 현재 상태 벡터 |
| **Key, Value** | 인코더 최종 출력 블록의 표상 |

⇒ 타겟 단어를 생성할 때 원문의 **어느 부분에 집중해야 하는지 동적으로 연결**합니다.

> 💡 **질의응답**
> **Q. 이전 레이어의 attention 결과가 하나의 벡터로 합쳐져 나오는데, 겹치는 문제는 없나?**
> A. 다음 블록/attention에서 새로운 $W^Q, W^K, W^V$를 독립적으로 곱하는 변환을 거쳐 **새로운 Q, K, V를 다시 생성**합니다. 겹치는 것은 걱정하지 않아도 됩니다.

---

## 10. 계산 복잡도 및 한계 (Quadratic Complexity)

### 한계

- 모든 단어 쌍의 상호작용($N \times N$)을 행렬 곱으로 일괄 계산하므로, 연산량과 메모리가 시퀀스 길이 $N$의 제곱에 비례합니다: **$O(N^2)$**
- RNN보다 **병렬 처리**는 매우 우수하지만, 수만 토큰 이상의 긴 문서에서는 메모리·계산 자원의 한계에 부딪힙니다.

### 개선 연구 방향

| 방향 | 내용 |
|---|---|
| **Quadratic computation 개선** | $QK^T$ 부분이 매우 크므로 $O(N^2)$을 $O(N)$으로 줄이려는 다양한 연구 |
| **Position representation 개선** | 위치 표현 방식 개선 |

### 그럼에도 불구하고

- $O(N^2)$을 줄이려는 연구가 많았지만, **실제 LLM들은 여전히 표준 $O(N^2)$ scaled dot-product attention을 사용**합니다.
- 2017년 논문 이후 숱한 변형이 나왔지만, 약간의 변경을 제외하면 **오리지널 Transformer의 뼈대가 여전히 지배적인 아키텍처**입니다.
- **Attention이 pretraining을 가능하게 만들었습니다.**

---
