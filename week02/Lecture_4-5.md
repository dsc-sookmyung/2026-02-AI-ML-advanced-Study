# CS224N Lecture 4-5 정리


# Lecture 4. Dependency Parsing

> 앞에서 배운 Word Vector와 Neural Network를 바탕으로,  
> **문장의 문법적 구조를 컴퓨터가 어떻게 분석하는가**를 다룬다.

## 1. Syntactic Structure

자연어 문장은 단순한 단어의 나열이 아니라, 단어 사이에 **문법적 관계**가 존재한다.

### Syntax

단어들이 어떻게 결합되어 문장을 구성하는지를 다루는 문법적 구조이다.

문장을 이해하려면 다음과 같은 관계를 파악해야 한다.

- 누가 행동했는가
- 무엇을 대상으로 행동했는가
- 어떤 단어가 다른 단어를 수식하는가



## 2. 문장 구조 표현 방법

### Constituency Structure

문장을 **구(Phrase) 단위**로 나누는 방식이다.

```text
The students love NLP

[S
   [NP The students]
   [VP love
       [NP NLP]
   ]
]
```

- `NP` : Noun Phrase
- `VP` : Verb Phrase

핵심은

> **어떤 단어들이 하나의 덩어리를 이루는가?**

에 초점을 둔다는 것이다.

### Dependency Structure

단어와 단어 사이의 관계를 직접 표현한다.

```text
students ──nsubj──> love
NLP      ──obj────> love
The      ──det────> students
```

즉,

> **누가 누구와 연결되어 있으며, 어떤 관계를 가지는가?**

를 표현한다.



## 3. Dependency Parsing

**Dependency Parsing**은 문장 속 단어들 사이의 의존 관계를 찾아 문장의 구조를 분석하는 방법이다.

```text
I love NLP.

     love
    /    \
   I     NLP
nsubj    obj
```

### Head와 Dependent

- **Head** : 다른 단어가 의존하는 중심 단어
- **Dependent** : Head에 의존하는 단어

일반적으로 각 단어는 하나의 Head를 가지며, 문장의 중심 단어는 `ROOT`와 연결된다.

### 대표적인 Dependency Relation

| Relation | 의미 | 예 |
|---|---|---|
| `nsubj` | 주어 | students → study |
| `obj` | 목적어 | NLP → study |
| `det` | 한정사 | the → student |
| `amod` | 형용사 수식 | good → student |
| `advmod` | 부사 수식 | quickly → run |

Dependency Parsing을 이용하면 단순히 단어 목록을 보는 것을 넘어, 문장 의미 분석이나 정보 추출 등에 필요한 구조를 얻을 수 있다.



## 4. Syntactic Ambiguity

하나의 문장이 여러 가지 구조로 해석될 수 있는 문제이다.

```text
I saw the man with a telescope.
```

가능한 해석:

1. 내가 망원경을 사용해서 남자를 보았다.
2. 망원경을 가지고 있는 남자를 보았다.

즉,

> **어떤 단어가 어떤 단어에 연결되는지에 따라 의미가 달라질 수 있다.**



## 5. Transition-based Dependency Parsing

문장 전체 구조를 한 번에 예측하지 않고, 여러 개의 작은 Action을 순차적으로 수행하여 Dependency Tree를 완성하는 방식이다.

Parser는 크게 다음 상태를 관리한다.

- **Stack** : 현재 분석 중인 단어
- **Buffer** : 아직 처리하지 않은 단어
- **Dependency Relations** : 지금까지 생성된 단어 간 관계

초기 상태 예시:

```text
Stack     : [ROOT]
Buffer    : [I, love, NLP]
Relations : []
```


## 6. Parser의 주요 Action

### SHIFT

Buffer의 첫 단어를 Stack으로 이동한다.

```text
Stack  : [ROOT]
Buffer : [I, love, NLP]

↓ SHIFT

Stack  : [ROOT, I]
Buffer : [love, NLP]
```

### LEFT-ARC

왼쪽 단어가 오른쪽 단어에 의존하도록 관계를 생성한다.

```text
love → I
```

### RIGHT-ARC

오른쪽 단어가 왼쪽 단어에 의존하도록 관계를 생성한다.



## 7. Neural Dependency Parser

Transition-based Parsing을 **분류 문제**로 볼 수 있다.

현재 Parser 상태를 보고 다음 Action을 예측한다.

```text
현재 Parsing State
        ↓
   Neural Network
        ↓
SHIFT / LEFT-ARC / RIGHT-ARC
```

Neural Network를 사용하면 단어, POS, Dependency 관련 정보를 Embedding으로 변환한 뒤 다음 Action을 학습할 수 있다.

```text
Words / POS / Dependency Features
             ↓
          Embedding
             ↓
        Neural Network
             ↓
           Softmax
             ↓
 SHIFT / LEFT-ARC / RIGHT-ARC
```



## 8. Greedy Parsing의 한계

Transition-based Parser는 현재 시점에서 가장 확률이 높은 Action을 선택하는 경우가 많다.

```text
SHIFT  : 0.7
LEFT   : 0.2
RIGHT  : 0.1

→ SHIFT 선택
```

이를 **Greedy Decision**이라고 한다.

하지만 초반에 잘못된 Action을 선택하면 이후 상태도 잘못되면서 오류가 계속 누적될 수 있다.

이를 **Error Propagation**이라고 한다.



## Lecture 4 핵심 정리

- Dependency Parsing은 단어 사이의 문법적 의존 관계를 Tree로 표현하는 방법이다.
- Transition-based Parser는 `Stack`, `Buffer`를 이용하여 `SHIFT`, `ARC` 등의 Action을 순차적으로 수행한다.
- Neural Dependency Parser는 다음 Action 선택을 Neural Network 기반 분류 문제로 학습한다.
- Greedy 방식은 빠르지만 초기 오류가 이후까지 이어질 수 있다.


# Lecture 5. RNN과 Language Model

> **Language Model → RNN → RNN 학습 → RNN의 한계**를 다룬다.

## 1. Language Model

**Language Model**은 지금까지 나온 단어들을 바탕으로 다음 단어의 확률을 예측하는 모델이다.

```text
the students opened their ___

books      0.35
laptops    0.20
eyes       0.10
door       0.01
```


쉽게 말하면:

> **앞 문맥을 보고 다음 단어를 예측하는 모델** 이다.

문장 전체 확률도 Chain Rule을 이용하여 계산할 수 있다.




## 2. 기존 N-gram 방식

RNN 이전에는 N-gram Language Model이 많이 사용되었다.

예를 들어 trigram은 이전 두 단어만 이용한다.

```text
I really love NLP
         ↑
     이전 2단어
```

### 한계

- 긴 문맥을 사용할 수 없다.
- 가능한 단어 조합이 너무 많아 Data Sparsity가 발생한다.

이러한 한계를 개선하기 위해 **RNN** 을 사용한다.



## 3. RNN

**RNN(Recurrent Neural Network)** 은 순서가 있는 데이터를 처리하기 위한 신경망이다.

이전 정보를 `Hidden State`에 저장하고, 다음 시점의 계산에 사용한다.

```text
현재 입력
   +
이전까지 기억한 정보
   ↓
새로운 Hidden State
```

### Hidden State

`Hidden State`는 이전까지의 문맥 정보를 기억하는 역할을 한다.

문장이 진행될수록 계속 업데이트된다.

```text
I → love → NLP
↓     ↓      ↓
h1 → h2 → h3
```



## 4. RNN Language Model

RNN을 Language Model에 적용하면 각 시점에서 다음 단어를 예측할 수 있다.

```text
Input Word
    ↓
Embedding
    ↓
RNN Hidden State
    ↓
Linear Layer
    ↓
Softmax
    ↓
다음 단어 확률
```

즉,

> 현재 단어와 이전 문맥을 함께 사용하여 다음 단어를 예측한다.



## 5. Weight Sharing

RNN은 모든 시점에서 **동일한 가중치(Parameter)** 를 사용한다.

이를 `Weight Sharing`이라고 한다.

### 장점

- 문장 길이가 달라도 처리할 수 있다.
- Sequence가 길어져도 Parameter 수가 크게 증가하지 않는다.
- 단어의 위치와 상관없이 동일한 패턴을 학습할 수 있다.



## 6. BPTT

RNN은 **Backpropagation Through Time(BPTT)** 을 이용해 학습한다.

시간축으로 펼쳐진 RNN을 거꾸로 따라가면서 Gradient를 계산한다.

```text
x1 → h1 → h2 → h3 → Loss
                       ↑
         ← ← ← Backpropagation
```



## 7. RNN의 한계

기본 RNN은 이론적으로 오래된 정보까지 기억할 수 있지만, 실제로는 긴 문장에서 먼 과거의 정보를 학습하기 어렵다.

이를 **Long-Term Dependency Problem** 이라고 한다.

```text
The students who attended the class several weeks ago
and studied very hard are ...
```

`are`를 올바르게 예측하려면 앞에 등장한 `students`가 복수형이라는 정보를 기억해야 한다.

하지만 두 단어 사이의 거리가 길어질수록 학습이 어려워질 수 있다.



## 8. Vanishing Gradient

BPTT 과정에서 Gradient가 반복적으로 곱해지면서 점점 작아지는 문제이다.

```text
Gradient
↓
점점 작아짐
↓
앞쪽까지 학습 신호가 전달되지 않음
```

결과적으로 오래전에 등장한 정보를 학습하기 어려워진다.

### 해결 방향

기본 RNN의 구조를 개선한

- **LSTM**
- **GRU**

등이 사용된다.



## 9. Exploding Gradient

반대로 Gradient가 반복적으로 곱해지면서 지나치게 커지는 문제이다.

Gradient가 너무 커지면 Parameter Update가 불안정해지고 학습이 어려워질 수 있다.

### 해결 방법

**Gradient Clipping**

Gradient가 일정 크기 이상 커지면 값을 제한하여 학습을 안정화한다.



## Lecture 5 핵심 정리

```text
Language Model
↓
앞 문맥을 이용하여 다음 단어 예측

N-gram
↓
긴 문맥과 Data Sparsity 문제

RNN
↓
Hidden State를 이용하여 이전 문맥 기억

BPTT
↓
시간축을 따라 역전파하여 학습

긴 Sequence
↓
Vanishing / Exploding Gradient 발생

Vanishing Gradient
↓
LSTM / GRU

Exploding Gradient
↓
Gradient Clipping
```

### Lecture 4 → Lecture 5 연결

```text
Lecture 4
Dependency Parsing
↓
문장의 문법적 구조를 분석

Lecture 5
Language Model + RNN
↓
단어의 순서와 앞 문맥을 이용하여
Sequence 자체를 학습
```