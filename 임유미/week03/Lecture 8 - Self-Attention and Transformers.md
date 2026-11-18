# Lecture 8 - Self-Attention and Transformers

## Lecture 8 - Self-Attention and Transformers

## Recap

<aside>

- 지금까지의 NLP 기본 전략: **양방향 LSTM**으로 인코딩 + **단방향 LSTM**으로 하나씩 디코딩 + **Attention**으로 유연하게 메모리 접근
- 이번 강의의 목표: 같은 문제를 풀되 **순환(recurrence)을 빼고 attention만으로** 구성한 새로운 building block(Self-Attention, Transformer)을 사용
</aside>

## 1. RNN의 두 가지 문제

### 선형 상호작용 거리(Linear Interaction Distance)

- RNN은 왼쪽→오른쪽으로 펼쳐지므로 **인접한 단어 사이의 상호작용은 쉽지만, 멀리 떨어진 단어는 상호작용하는 데 오래 걸림**
    - 예) “the chef who ... was” 에서 chef와 was 사이에 RNN 연산이 시퀀스 길이만큼 필요
- LSTM이 단순 RNN보다 기울기 전달을 잘하지만 완벽하지 않음
- 관련 있는 단어끼리는 거리와 상관없이 **직접 연결**되길 원함

### 병렬화 불가능 (Dependence on Time)

- RNN의 forward/backward pass는 시퀀스 길이 O(T)만큼의 **병렬화 불가능한 연산**이 필요
- t=5의 은닉 상태를 계산하려면 t=3, t=4가 먼저 계산돼야 함 → GPU를 효율적으로 못 씀

### Attention의 해결방법

- 각 단어를 한 번의 큰 연산으로 모든 단어를 보며 표현 구성, **모든 단어에 대해 동시에(병렬) 계산 가능**
- 단어 사이 거리와 관계없이 상호작용 가능 → **두 문제를 모두 해결**

## 2. Self-Attention

### 개념: Key-Value 저장소에서의 퍼지 조회

- 일반 lookup table(파이썬 dict): query가 key 하나와 **정확히 일치** → value 반환
- Attention: query가 **모든 key와 부드럽게(soft) 일치** → 유사도로 가중치를 매겨 value들의 **가중합** 반환

### 수식

<aside>

1. 단어 임베딩 `x_i = E w_i` (문맥 무관, 자기 자신에만 의존)
2. 세 개의 가중치 행렬로 변환 (`Q, K, V ∈ R^(d×d)`)
    - `q_i = Q x_i`, `k_i = K x_i`, `v_i = V x_i`
3. 모든 쌍의 유사도(점수): `e_ij = q_i^T k_j`
4. softmax로 가중치 정규화: `α_ij = exp(e_ij) / Σ_j' exp(e_ij')`
5. 출력: `output_i = Σ_j α_ij v_j` (value의 가중합)
</aside>

- 모든 단어가 query, key, value 세 역할을 모두 수행함

## 3. Self-Attention을 대체 블록으로 쓰기 위한 3가지 문제와 해결

### ① 순서 정보가 없음

- Self-attention은 **집합(set)에 대한 연산** → “Zuko made his uncle”와 “his uncle made Zuko”가 **똑같은 표현**이 됨 (임베딩이 위치에 의존하지 않기 때문)
- 해결: 각 위치 인덱스를 벡터 `p_i`로 표현해 **입력 임베딩에 더함** (`x_i + p_i`), 네트워크 입력에서 한 번만 하면 충분

| 방법 | 설명 | 장점 | 단점 |
| --- | --- | --- | --- |
| **Sinusoidal** | 차원마다 주기가 다른 sin/cos 사용 | 주기성 덕분에 긴 시퀀스로 외삽 가능할 것 같다는 직관 | 실제로는 외삽이 잘 안 됨 |
| **Learned** | `d × n` 행렬을 파라미터로 학습 | 데이터에 맞춰 자유롭게 학습, 현재 대부분 사용 | **최대 길이 n을 넘는 입력은 처리 불가** (에러 발생) |
- 그 외: 상대적 위치(relative linear position), 의존 구문 트리 기반 표현 등
- “p가 위치를 표현하는 건 어떻게 아나?” → p 행렬은 항상 같은 인덱스에 붙으므로 **위치와만 상관관계**가 있어 암묵적으로 학습됨
- 시퀀스 길이 제한은 지금도 문제: self-attention이 시퀀스 길이에 **제곱(n²) 복잡도**이기 때문 (큰 모델도 ~4,000 토큰 수준)

### ② 비선형성이 없음

- Self-attention을 반복해도 결국 value 벡터의 **가중 평균만 계속 반복**하는 셈
- 해결: 각 출력 벡터에 **Feed-forward Network(MLP)** 를 독립적으로 적용 (position-wise)
    - `행렬 곱 → 비선형 함수 → 행렬 곱`, 보통 hidden layer 1개
    - 연산을 효율적이고 병렬적으로 쌓을 수 있어 GPU에 유리

### ③ 미래를 보면 안 됨 (Masking)

- 언어 모델링, 기계 번역의 디코더에서는 **미래 단어를 보고 예측하면 안 됨**
- 매 시점마다 key/query/value를 따로 정의하면 병렬화가 안 되므로, **n×n 전체를 계산한 뒤 미래 위치를 마스킹**
    - `e_ij = q_i^T k_j` (j ≤ i), `∞` (j > i) → softmax 후 가중치 0
- **디코더에는 마스킹 O, 인코더에는 마스킹 X** (양방향 LSTM vs 단방향 LSTM과 같은 구분)
- 학습 시 미래 정답을 알려주면 배울 게 없어서 마스킹이 필요함

<aside>

**Minimal self-attention architecture** = Self-attention + Position embedding + Feed-forward(비선형성) + Masking → 이를 여러 번 반복

하지만 실제로는 이 구조를 그대로 쓰지 않음 → Transformer의 추가 요소 필요

</aside>

## 4. Transformer

- Transformer가 연구의 종착점은 아님. self-attention의 문제와 개선 방향을 계속 고민해야 함

### Multi-Head Self-Attention

- 단어를 볼 때 **여러 이유로 서로 다른 곳을 봐야 함** (예: 개체/고유명사 정보, 문법적으로 관련된 단어)
- 하나의 attention으로 모두 평균내기는 어려움 → **head 여러 개**를 둠

#### Sequence-stacked(행렬) 형태

- 단어 임베딩을 쌓아 `X ∈ R^(n×d)`, `XQ`, `XK`, `XV`를 한 번의 큰 행렬 곱으로 계산 (벡터화)
- `output = softmax(XQ (XK)^T) · XV` → n×n 행렬 × n×d 행렬

#### Multi-head

- head 수 h개, 각 head마다 독립적인 `Q_l, K_l, V_l ∈ R^(d × d/h)` (저차원으로 투영)
- 각 head가 독립적으로 attention 수행 → 출력을 **concat** → 마지막 **선형 변환**으로 섞음
- 구현은 `XQ`를 `(n, h, d/h)`로 **reshape 후 head 축을 batch처럼** 사용 → 비용이 늘지 않음

### Scaled Dot Product Attention

- `d`가 커지면 내적 값이 커져 softmax 입력이 매우 커지고 **기울기가 작아짐**
- 해결: 점수를 `√(d/h)`로 **나눠서** 초기화 시 값이 커지지 않게 함
- 구현이 쉬워 이후 항상 사용한다고 가정

### Residual Connection

- `layer 출력 + layer 입력` → 이전 층 대비 **잔차(residual)만 학습**하면 됨
- 기울기가 항등 경로를 통해 1로 전달 → 기울기 소실 완화
- 초기화 시 전체가 항등 함수처럼 동작해 좋은 출발점이 됨 (loss landscape가 더 매끄러워짐)

### Layer Normalization

- 층 내부의 **불필요한 변동(uninformative variation)** 을 줄여 학습을 빠르게 함
- 단어 벡터 `x ∈ R^d` 하나에 대해 평균, 표준편차로 정규화

```jsx
μ = (1/d) Σ_j x_j`, `σ = √((1/d) Σ_j (x_j − μ)²)

output = (x − μ) / (σ + ε) ⊙ γ + β
```

(ε: 0으로 나누는 것 방지, γ·β: 학습 가능한 scale/shift로 선택 사항)

- **통계는 각 단어마다 독립적**으로 계산 (시퀀스 내 다른 단어, 배치 내 다른 샘플과 공유 X)
    - 배치 전체를 공유하는 Batch Norm의 단점(관련 없는 샘플에 forward가 의존)을 해결하려고 만들어짐

### Transformer 구조

#### Decoder

- 한 block = **Masked Multi-Head Self-Attention → Add & Norm → Feed-Forward → Add & Norm**
- 이 block을 여러 번 반복 (언어 모델 등에 사용)

#### Encoder

- Decoder와 거의 동일하지만 **마스킹을 하지 않음** (양방향 문맥)

#### Encoder-Decoder (원 논문 *Attention Is All You Need*)

- 인코더: 마스킹 없는 multi-head attention으로 소스 문장 처리
- 디코더: Masked self-attention + **Cross-Attention** 추가
    - **Query = 디코더 출력 z**, **Key, Value = 인코더 출력 h**
    - 디코더의 모든 단어가 인코더의 모든 단어를 참조

| 구조 | 마스킹 | 용도 |
| --- | --- | --- |
| Encoder | X (양방향) | 입력 이해, 분석 |
| Decoder | O (미래 차단) | 텍스트 생성, 언어 모델 |
| Encoder-Decoder | 인코더 X, 디코더 O + Cross-Attention | 기계 번역 등 |

#### 가변 길이 입력

- GPU에는 일정 길이로 입력 → **pad 토큰**으로 채우고, pad 위치는 점수를 `∞`로 마스킹
- 모든 입력이 짧으면 최대 길이 대신 더 작은 길이로 설정해 계산 절약 가능

## 5. Transformer의 성과와 한계

### 성과

- 기계번역에서 기존 시스템보다 극적으로 좋진 않았지만 **학습 효율이 훨씬 높았음** (병렬화로 더 많은 데이터를 빠르게 처리)
- 문서 생성 등에서 LSTM seq2seq을 대체, **사전학습(pre-training) 혁명**을 가능하게 함 (다음 강의)
- 방대한 데이터·연산을 활용할 수 있어 NLP 거의 모든 최신 발전에서 LSTM을 압도

### 한계와 연구 방향

- **이차(Quadratic) 연산량**: 모든 쌍의 상호작용 때문에 block당 연산이 시퀀스 길이의 제곱으로 증가
    - RNN은 선형으로 증가했으므로 이 점에서는 퇴보
    - 긴 문서/소설 처리가 어려움 → 저차원 매핑 등 개선 시도
    - 다만 GPT-3, ChatGPT 같은 큰 모델에서는 연산 대부분이 self-attention이 아니라서 꼭 필요한지는 연구 중
- 더 나은 **위치 표현** 방법 (절대 인덱스가 최선이 아닐 수 있음)
- 수많은 변형이 나왔지만 **원래 Transformer + 소수의 수정**이 여전히 거의 최고
    - 예) feed-forward의 비선형 함수 교체는 중요했고 효과가 오래 지속됨