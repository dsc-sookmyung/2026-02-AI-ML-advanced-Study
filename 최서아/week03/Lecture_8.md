# CS224N Lecture 8 정리

> 앞 강의에서 학습한 **RNN / LSTM / Seq2Seq / Attention**을 바탕으로  
> **Self-Attention → Query·Key·Value → Multi-Head Attention → Positional Encoding → Transformer**로 확장한다.

## 1. RNN과 LSTM의 한계

기존 **RNN / LSTM**은 단어를 순서대로 처리하므로 이전 시점의 계산이 끝나야 다음 시점으로 넘어갈 수 있다.

```text
x1 → x2 → x3 → x4
 ↓    ↓    ↓    ↓
h1 → h2 → h3 → h4
```

- **순차 처리**: 시점 간 의존성으로 병렬화에 제약이 있다.
- **Long-Term Dependency**: 먼 위치에 있는 단어 사이의 정보를 전달하기 어렵다. LSTM은 이를 완화하지만 완전히 해결하지는 못한다.

**Transformer**는 RNN의 순차적인 Hidden State 전달 대신 **Attention**을 중심으로 토큰 간 관계를 계산한다.

## 2. Self-Attention

**Self-Attention**은 같은 Sequence에 있는 각 토큰이 **다른 토큰을 얼마나 참고할지 계산**하여, 문맥이 반영된 새로운 표현을 생성하는 방법이다.

예를 들어 다음 문장을 보자.

```text
The animal didn't cross the street because it was tired.
```

`it`의 의미를 파악하려면 앞의 `animal`과 같은 단어를 참고해야 한다.

```text
입력 Sequence
      ↓
각 토큰 사이의 관련성 계산
      ↓
중요한 토큰의 정보를 더 크게 반영
      ↓
문맥이 반영된 토큰 표현
```

### 기존 Attention과 Self-Attention 비교

| 기존 Encoder-Decoder Attention | Self-Attention |
|---|---|
| Decoder가 Encoder의 출력 정보를 참고 | 같은 Sequence 안의 토큰들이 서로 참고 |
| 입력과 출력 사이의 관계 학습 | 입력 내부 또는 출력 내부의 관계 학습 |
| Seq2Seq 번역 등에 사용 | Transformer의 핵심 연산 |

## 3. Query, Key, Value (Q, K, V)

Self-Attention은 각 입력 토큰의 표현을 학습 가능한 행렬로 변환해 세 가지 벡터를 만든다.

- **Query(Q)**: 현재 토큰이 찾으려는 정보
- **Key(K)**: 각 토큰이 어떤 정보와 관련 있는지 판단하는 기준
- **Value(V)**: 실제로 전달할 정보

```text
입력 X
 ├─ W_Q → Query(Q)
 ├─ W_K → Key(K)
 └─ W_V → Value(V)
```

### Self-Attention 계산 과정

**① Query와 Key의 유사도 계산**

```text
현재 토큰의 Query
       ↓
모든 토큰의 Key와 비교
       ↓
Attention Score
```

**② Softmax로 Attention Weight 계산**

관련성 점수를 정규화해 어떤 토큰에 더 집중할지 결정한다.

**③ Value를 가중합**

Attention Weight가 큰 토큰의 Value를 더 많이 반영하여 새로운 표현을 만든다.

### Scaled Dot-Product Attention

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $QK^T$: 토큰 간 관련성 점수
- $\sqrt{d_k}$: 점수가 지나치게 커지는 현상 완화
- `softmax`: Attention Weight 계산
- $V$: Weight를 반영하여 최종 정보 생성

> **Q와 K로 관련성을 계산하고, 그 결과를 V에 적용한다.**

## 4. Multi-Head Attention

**Multi-Head Attention**은 Self-Attention을 여러 개의 Head에서 병렬로 수행하는 구조이다.

```text
          입력 X
             ↓
    ┌────────┼────────┐
    ↓        ↓        ↓
  Head 1   Head 2   Head 3
    └────────┼────────┘
             ↓
        Concatenate
             ↓
      Linear Projection
             ↓
           Output
```

각 Head는 서로 다른 가중치를 사용하므로 다양한 관계를 학습할 수 있다.

- 주어와 동사의 관계
- 단어 간 의미적 연관성
- 멀리 떨어진 단어 사이의 관계

단, **각 Head가 반드시 특정 관계 하나만 학습하도록 지정되는 것은 아니다.**

## 5. Positional Encoding

Self-Attention만으로는 토큰의 **순서 정보**를 충분히 표현할 수 없다.

```text
Dog bites man.
Man bites dog.
```

같은 단어를 사용하더라도 순서에 따라 의미가 달라진다.

따라서 Transformer는 토큰의 위치 정보를 추가한다.

```text
Token Embedding
       +
Positional Encoding
       ↓
Transformer 입력
```

> **Positional Encoding**: 순차적으로 처리하지 않아도 토큰의 위치와 순서를 구분하도록 돕는 정보

원래 Transformer에서는 `sin`, `cos` 함수를 활용한 위치 인코딩을 사용한다.

## 6. Transformer Encoder

**Encoder**는 입력 Sequence를 받아 각 토큰의 문맥적 표현을 생성한다.

```text
  Token Embedding + Position
              ↓
    Multi-Head Self-Attention
              ↓
       Add & LayerNorm
              ↓
     Feed Forward Network
              ↓
       Add & LayerNorm
              ↓
       Encoder Output
```

### 주요 구성 요소

- **Self-Attention**: 입력 토큰 사이의 관계 계산
- **Feed Forward Network(FFN)**: 각 토큰의 표현을 비선형 변환
- **Residual Connection(Add)**: 입력을 Sublayer 출력에 더해 정보와 Gradient 전달을 도움
- **Layer Normalization**: 학습을 안정화

위 구조는 원래 Transformer의 Encoder Layer를 간략화한 것이다.

## 7. Transformer Decoder

**Decoder**는 이전에 생성된 토큰과 Encoder의 정보를 참고하여 다음 토큰을 생성한다.

```text
  Output Embedding + Position
              ↓
   Masked Self-Attention
              ↓
   Encoder-Decoder Attention
              ↓
      Feed Forward Network
              ↓
       Linear + Softmax
              ↓
         다음 토큰
```

원래 Transformer Decoder에는 각 Sublayer마다 Residual Connection과 Layer Normalization도 적용된다.

### Masked Self-Attention

다음 단어를 예측할 때 **미래 토큰을 미리 참고하지 못하도록** 가리는 방식이다.

```text
           I  love  NLP
I          O    X    X
love       O    O    X
NLP        O    O    O

O: 참고 가능 / X: 참고 불가
```

이를 **Causal Masking**이라고 한다.

> Decoder는 자기 자신과 **이전 토큰만 참고**해 다음 토큰을 예측할 수 있다.

## 8. RNN / LSTM과 Transformer 비교

| RNN / LSTM | Transformer |
|---|---|
| Hidden State를 순서대로 전달 | Self-Attention으로 토큰 간 관계 계산 |
| 순차적 처리에 의존 | 학습 시 입력 토큰 병렬 처리에 유리 |
| 먼 토큰 사이의 정보 전달 경로가 김 | Self-Attention으로 먼 토큰도 직접 참고 가능 |
| 처리 과정 자체에 순서가 반영됨 | 별도의 위치 정보가 필요 |

단, 기본 Self-Attention의 연산량과 메모리 사용량은 **Sequence 길이에 대해 제곱에 비례**하는 한계가 있다.

## Lecture 8 핵심 흐름

```text
RNN / LSTM
↓
순차 처리의 제약
↓
Self-Attention
↓
Q와 K로 관련성 계산 → V의 가중합
↓
Multi-Head Attention
↓
여러 관점에서 토큰 관계 학습
↓
Positional Encoding
↓
순서 정보 추가
↓
Transformer Encoder / Decoder
```

## 핵심 개념 정리

- **Self-Attention**: 같은 Sequence 안의 토큰들이 서로의 정보를 참고하여 표현을 업데이트
- **Query(Q)**: 어떤 정보를 찾을지 나타내는 벡터
- **Key(K)**: Query와 비교하여 관련성을 계산하는 벡터
- **Value(V)**: Attention Weight를 적용할 정보 벡터
- **Scaled Dot-Product Attention**: $QK^T$를 $\sqrt{d_k}$로 나눈 뒤 Softmax와 Value를 적용
- **Multi-Head Attention**: 여러 Attention을 병렬로 수행하여 다양한 관계 학습
- **Positional Encoding**: 토큰의 위치 및 순서 정보 제공
- **Encoder**: 입력 토큰의 문맥적 표현 생성
- **Decoder**: Masked Attention 등을 통해 다음 토큰 생성
- **Causal Masking**: 미래 토큰을 참조하지 못하도록 차단

---

## Lecture 6 → Lecture 8 연결

```text
Lecture 6
LSTM
↓
Encoder-Decoder / Seq2Seq
↓
고정 Context Vector의 정보 병목
↓
Attention으로 Encoder의 여러 상태 참고

Lecture 8
Self-Attention으로 Sequence 내부 관계 계산
↓
Multi-Head Attention + Positional Encoding
↓
Transformer
```
