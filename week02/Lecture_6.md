# CS224N Lecture 6 정리

> 앞 강의의 **RNN / Language Model**을 이어서  
> **LSTM → Encoder-Decoder → Seq2Seq → Machine Translation → Attention**으로 확장한다.



## 1. LSTM

**LSTM(Long Short-Term Memory)** 은 기본 RNN의 **Long-Term Dependency**와 **Vanishing Gradient** 문제를 완화하기 위해 등장한 구조이다.

기본 RNN과 달리 `Cell State`를 추가하여 중요한 정보를 더 오래 유지한다.

### Cell State

`Cell State`는 장기 기억 저장소 역할을 한다.

```text
C(t-1) ==========================> C(t)
             ↑        ↑
           삭제      추가
```

기본 RNN처럼 매 단계에서 정보를 완전히 새로 계산하는 것이 아니라,

- 필요한 과거 정보는 유지하고
- 필요 없는 정보는 삭제하고
- 새로운 정보는 추가

하면서 상태를 업데이트한다.



## 2. LSTM의 Gate

LSTM은 **Gate**를 이용하여 어떤 정보를 유지하거나 삭제할지 결정한다.

Gate는 보통 `sigmoid`를 사용하며, 출력값은 `0~1` 사이이다.

```text
0 → 거의 차단
1 → 거의 전부 통과
```

### Forget Gate

기존 Cell State에서 **어떤 정보를 버릴지 결정**한다.

> 필요 없는 과거 정보를 삭제하는 역할



### Input Gate

현재 입력과 이전 문맥을 이용하여 **새롭게 저장할 정보를 결정**한다.

```text
현재 입력
+
이전 문맥
↓
새로운 후보 정보 생성
↓
필요한 정보만 저장
```

Cell State는 다음 개념으로 업데이트된다.

```text
새 Cell State
=
남겨둘 과거 정보
+
새로 추가할 정보
```

즉,

> **Forget Gate + Input Gate → 새로운 장기 기억 생성**



### Output Gate

현재 Cell State 중 **현재 시점에서 필요한 정보만 선택하여 Hidden State로 출력**한다.

```text
Cell State
   ↓
필요한 정보 선택
   ↓
Hidden State h(t)
```



## 3. LSTM 전체 흐름

```text
과거 기억 C(t-1)
        ↓
① Forget Gate
   필요 없는 정보 삭제
        ↓
② Input Gate
   새로운 정보 추가
        ↓
새로운 기억 C(t)
        ↓
③ Output Gate
   현재 필요한 정보 출력
        ↓
Hidden State h(t)
```

즉, LSTM은

> **버릴 것 → 새로 기억할 것 → 출력할 것**

을 각각 따로 학습한다.

### 기본 RNN과 LSTM 비교

| 기본 RNN | LSTM |
|---|---|
| Hidden State 중심 | Hidden State + Cell State |
| 기억 제어 기능 없음 | Gate를 통해 기억 제어 |
| 긴 문맥 학습 어려움 | 긴 문맥 학습 개선 |
| Vanishing Gradient 문제 큼 | 문제 완화 |
| 구조 단순 | 구조 복잡 |



## 4. Sequence-to-Sequence

**Seq2Seq(Sequence-to-Sequence)** 는 하나의 Sequence를 입력받아 다른 Sequence를 출력하는 모델이다.

대표적인 활용 예시는 **Machine Translation**이다.

```text
English Sentence
       ↓
     Seq2Seq
       ↓
Korean Sentence
```

입력과 출력의 길이가 서로 다를 수 있으므로, 입력을 읽는 부분과 출력을 생성하는 부분을 분리한다.

이를 **Encoder-Decoder 구조**라고 한다.


## 5. Encoder

Encoder는 입력 Sequence를 순서대로 읽는다.

```text
x1 → h1
      ↓
x2 → h2
      ↓
x3 → h3
```

마지막 Hidden State에는 입력 문장의 정보가 압축되어 들어간다.

이를 **Context Vector**라고 한다.

> Context Vector = 입력 문장을 압축한 표현


## 6. Decoder

Decoder는 Encoder가 생성한 Context Vector를 받아 출력 Sequence를 생성한다.

```text
Encoder
I → love → NLP
          ↓
      Context Vector
          ↓
       Decoder
          ↓
나는 → NLP를 → 좋아한다
```

Decoder는 문장을 한 번에 만드는 것이 아니라, **한 단어씩 다음 단어를 예측하면서 생성**한다.

```text
나는
↓
다음 단어 예측

NLP를
↓
다음 단어 예측

좋아한다
```


## 7. Seq2Seq 전체 구조

```text
Input Sequence

x1 → x2 → x3 → x4
 ↓    ↓    ↓    ↓
Encoder RNN / LSTM
               ↓
        Context Vector
               ↓
Decoder RNN / LSTM
 ↓    ↓    ↓    ↓
y1 → y2 → y3 → y4

Output Sequence
```



## 8. Machine Translation

예를 들어 다음 문장을 번역한다고 하자.

```text
Input:
He likes coffee
```

### Encoder

```text
He
↓
h1

He likes
↓
h2

He likes coffee
↓
h3
```

마지막 `h3`에 입력 문장의 정보가 담긴다.

### Decoder

```text
그는
↓
커피를
↓
좋아한다
↓
<END>
```

### 특별한 Token

- `<START>` : 문장 생성을 시작하는 신호
- `<END>` : 문장 생성을 종료하는 신호


## 9. Training과 Inference

### Training

학습할 때는 정답 문장을 알고 있으므로 **실제 이전 정답 단어를 입력으로 넣고 다음 단어를 예측**한다.

```text
입력 <START>
정답 나는

입력 나는
정답 NLP를

입력 NLP를
정답 좋아한다

입력 좋아한다
정답 <END>
```

이러한 방식을 **Teacher Forcing**이라고 한다.

### Inference

실제 예측 시에는 정답이 없으므로, **이전에 모델이 예측한 단어를 다시 다음 입력으로 사용**한다.

```text
<START>
↓
나는
↓
NLP를
↓
좋아한다
↓
<END>
```

## 10. Seq2Seq의 한계

기본 Seq2Seq에서는 Encoder가 입력 전체를 **하나의 고정된 Context Vector**에 압축해야 한다.

```text
긴 입력 문장
↓
하나의 Context Vector
```

문장이 길어질수록 다음과 같은 다양한 정보를 하나의 Vector에 모두 담기 어려워진다.

- 주어
- 목적어
- 시제
- 수식 관계
- 단어 의미
- 전체 문맥

이를 **Information Bottleneck** 문제라고 볼 수 있다.


## 11. Attention 등장

Seq2Seq의 정보 손실 문제를 줄이기 위해 **Attention**이 등장한다.

기존 방식은 Encoder의 마지막 Hidden State 하나만 이용한다.

```text
h1 → h2 → h3 → h4
               ↓
        Context Vector
               ↓
            Decoder
```

Attention에서는 Decoder가 필요할 때마다 **Encoder의 모든 Hidden State를 참고**할 수 있다.

```text
h1   h2   h3   h4
 ↑    ↑    ↑    ↑
 └──── Decoder가 필요할 때 참고
```

즉,

> 하나의 Vector에 모든 정보를 압축하지 않고, 출력할 때 필요한 입력 위치의 정보를 직접 참고한다.

이를 통해 긴 문장에서 발생하는 정보 손실을 줄일 수 있다.


## Lecture 6 핵심 흐름

```text
기본 RNN
↓
긴 문맥 학습 어려움
↓
LSTM
↓
Cell State + Gate로 기억 관리

LSTM
↓
Encoder / Decoder 구조
↓
Seq2Seq
↓
Machine Translation

기본 Seq2Seq
↓
하나의 Context Vector에 모든 정보 압축
↓
Information Bottleneck
↓
Attention 등장
```

## 핵심 개념 정리

- **LSTM** : 긴 정보를 더 잘 기억하도록 개선한 RNN
- **Cell State** : 장기 기억을 저장하는 상태
- **Forget Gate** : 필요 없는 과거 정보 삭제
- **Input Gate** : 새로운 정보 저장
- **Output Gate** : 현재 필요한 정보 출력
- **Seq2Seq** : 하나의 Sequence를 다른 Sequence로 변환
- **Encoder** : 입력 Sequence를 읽어 표현을 생성
- **Context Vector** : 입력 문장을 압축한 Vector
- **Decoder** : Context를 이용해 출력 Sequence 생성
- **Teacher Forcing** : 학습 시 실제 이전 정답을 Decoder 입력으로 사용
- **Information Bottleneck** : 긴 입력 정보를 하나의 Vector에 압축하면서 발생하는 한계
- **Attention** : Encoder의 여러 Hidden State 중 필요한 정보를 직접 참고

---

## Lecture 5 → Lecture 6 연결

```text
Lecture 5
RNN
↓
Hidden State로 이전 문맥 기억
↓
Vanishing Gradient / Long-Term Dependency 문제

Lecture 6
LSTM
↓
Cell State와 Gate로 기억 관리
↓
Encoder-Decoder
↓
Seq2Seq
↓
Attention
```