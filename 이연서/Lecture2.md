## 1. 최적화 기법: SGD와 Negative Sampling

### 기존 Gradient Descent의 문제

Lecture1에서 배운 내용

: Loss를 계산하고 → Gradient를 구하고 → 벡터를 수정한다

학습 데이터가 적을 경우 전체를 다 계산 후 Gradient 값을 구해도 문제가 없다

⇒ 실제 corpus에 수십억개의 word window가 존재한다고 생각해보면

수십억개 전부 계산 → 전체 Loss 계산 → 전체 Gradient 계산 → 드디어 벡터 한번 수정..

이라는 과정을 거쳐야 한다… 그래서 **너무 느리다 !!!** (벡터 한번 바꾸려고 수십억개를 봐야 함)

### SGD(Stochastic Gradient Descent)의 등장

SGD : 전체 학습 데이터의 gradient를 한번에 계산하지 않고 일부 sample을 이용해 gradient를 근사하고, 파라미터를 자주 업데이트하는 방법

SGD 아이디어 : 전체 데이터를 다 보고 한번 수정하는 대신, 조금 보고 바로바로 수정하자

즉, 기존 GD가 전체 데이터를 다 보고 벡터 한번 수정했다면 **SGD는 다 보지 말고 일부만 본 다음에 수정하자 ~** 는 아이디어이다

따라서 GD보다 훨씬 자주 업데이트 할 수 있다는 장점이 존재

#### SGD 원리

쉽게 말하면 Gradient Descent를 현실적으로 빠르게 돌리기 위한 방식

- 기본 Gradient Descent : 전체 학습 데이터를 다 보고 gradient를 계산한 뒤 한번 업데이트
- SGD : 전체 학습 데이터를 다 보느느 대신, 일부 데이터만 보고 gradient를 계산한 뒤 바로 업데이트

+) `Stochastic` = 확률적, 무작위적

→ 따라서 SGD는 전체 데이터를 항상 순서대로 다 계산하는 것이 아니라 학습 데이터에서 sample/window를 뽑아 gradient를 계산하기 때문에 업데이트 방향에 약간의 noise가 발생하긴 한다 . . . 그래도 한번한번 계산량이 훨씬 적기때문에 전체적으로는 빠르게 학습이 가능하다

### SGD가 가진 문제점

`Word2Vec`의 `Softmax`를 생각해보면

![스크린샷 2026-09-29 오후 3.34.13.png](Lecture%202/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-09-29_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.34.13.png)

`center`가 `brown`일 경우 `brown ↔ fox`만 계산하면 된다

그러나 `Vocavulary`가 100만개라면 ? ⇒ 100만개 단어 전부와 내적(dot product)를 계산해야 한다

SGD로 `window` 하나만 골랐어도, 그 `window` 하나를 계산하기 위해 `voacabulary` 전체를 훑어야 하는 문제점이 발생한다 !!

### Negative Sampling

- 기존 `softmax` : brown이 주어졌을 때 vocabulary의 모든 단어 중 fox가 나올 확률은 얼마인가?
- `Negative Sampling` : `"brown - fox"`는 진짜 같이 등장한 pair인가?

```
1. (brown, fox) → ✅ TRUE
2. (brown, banana) → ❌ FALSE
3. (brown, airplane) → ❌ FALSE
4. (brown, doctor) → ❌ FALSE

# 학습 과정
1. brown ↔ fox => score ↑
2. brown ↔ banana => score ↓
3. brown ↔ airplane => score ↓
4. brown ↔ doctor => score ↓
```

위의 방식을 사용하면 100만 단어와 비교할 필요가 없다 !!!

⇒ negative sample 5개만 뽑아서 `1개의 positive + 5개의 negative`만 계산하면 된다~

#### Negative Sampling 원리

방향성 : `positive pair`의 점수는 높이고, `negative pair`에는 점수는 낮추는 방향

(= positive label = 1, negative label = 2를 주는 느낌이라고 이해)

→ 모델이 각 pair의 실제 예측값을 1 또는 0에 가깝게 만들도록 `vector`를 조정한다

⇒ 즉, 진짜 같이 나온 단어는 가깝게 & 랜덤으로 뽑은 가짜 단어는 멀리있도록 `vector`를 조정한다 ~ (수학적 : `dot product`를 크게/작게 만든다)

`Negative Sampling`은 일종의 대비 학습 느낌이다

→ 원래 방식인 `softmax`는 `center word`와 `vocabulary` 전부를 비교를 매번 수행했다면 `Negatvie Sampling`은 100만개 전부는 너무 비싸니까(계산량이 많으니까) 진짜 1개 + 가짜 몇 개만 골라서 비교하자 ~ 는 아이디어 !!!

+) 이 값을 그냥 두는게 아니라 `Sigmoid`를 사용해서 0 ~ 1 사이의 값으로 바꿔줘야 한다 !!! (그래야 확률값이 됨)

- 진짜 `outside word` → `sigma(u_o^Tv_c)`를 크게
- `negative sample` → `sigma(-u_k^Tv_c)`를 크게

### SGD vs. Negative Sampling

둘 다 계산을 줄이지만 무엇을 줄이는지가 다르다

- `SGD` : 얼마나 많은 **학습데이터**를 볼 지
  - 전체 corpus X → 일부 window만
- `Negative Samplin`g : **한 window에서 vocabulary**를 얼마나 많이 비교할지
  - 전체 vocabulary X → positive + 몇개의 negative

그래서 둘을 같이 사용하면 계산량을 아주 많이 줄일 수 있다!!

```
거대한 corpus

↓ SGD

일부 window만 가져옴

↓ Negative Sampling

그 window에서도
모든 vocabulary와 비교하지 않고
positive + 몇 개의 negative만 계산

↓

빠르게 업데이트
```

## 2. 카운트 기반 모델과 GloVe (Global Vectors)

### Co-occurrence Matrix

- `Co-occurrence` = 공동 등장
  즉, 두 단어가 같은 문맥 안에서 몇 번 같이 등장했는지 카운팅 하는 것
  ```
  I like coffee.
  I drink coffee.
  I like tea.
  I drink tea.
  ```
  ```
  # coffe의 주변 단어
  I
  like
  drink

  # tea의 주변 단어
  I
  like
  drink
  ```

**Co-occurrence Matrix**

| target word | like | drink | car | road |
| ----------- | ---- | ----- | --- | ---- |
| coffee      | 10   | 8     | 0   | 0    |
| tea         | 9    | 9     | 0   | 0    |
| car         | 0    | 0     | 12  | 15   |

- `coffee`의 row : [10, 8, 0, 0]
- `tea`의 row : [9, 9, 0, 0]

⇒ 둘이 상당히 비슷함을 확인할 수 있다

⇒ 따라서 주변 단어들과 함께 등장한 횟수를 벡터로 사용하면 비슷한 의미를 가진 단어들이 비슷한 벡터를 가질 수 있겠다 ~ 라는 아이디어에서 출발한 것!

### Co-occurrence vs. Word2Vec

- `Word2Vec` : 주변 단어를 예측하면서 벡터를 학습 (ex. brown → fox가 나올까?) ⇒ predictive method
- `Co-occurrence` : 실제 등장 횟수를 직접 카운트 (ex. coffee와 drink는 실제로 몇번 같이 나왔지?) ⇒ Count-based method

### Co-occurrence Matrix를 단독으로 사용했을 때의 문제점

Vacabulary가 100만 단어일 경우 matrix의 크기가 1,000,000 X 1,000,000 가된다

게다가 대부분의 단어는 서로 같이 등장하지 않는다 (일부 단어쌍만 같이 많이 등장)

⇒ 즉 값이 0인 부분이 많아진다 (해당 벡터를 `Sparse Vector`라고 한다 ex. `[0, 0, 0, 15, 0, 0, 27, 0, …]`)

### SVD의 등장

아이디어 : 엄청 큰 sparse matrix를 중요한 정보는 최대한 유지하면서 작은 dense vector로 압축하자

coffee = `[12, 0, 0, 47, 0, 18, 0, 0, ... 100만 차원]`

- `SVD` 적용 → coffee = `[0.31, -0.72, 0.18, 0.55, ... 100 ~ 300차원]`

이를 통해 `Dense Word Vector`를 만들 수 있다!!

흐름 정리) `Corpus` → 단어들이 공동으로 등장한 횟수 카운트 → `Co-occurrence Matrix` 계산 → `SVD` 적용 → 작은 `Dense Word Vector` 도출

### SVD의 한계

SVD를 사용하면 벡터는 얻을 수 있지만

- matrix 자체가 너무 크고
- 새로운 데이터가 추가되면 matrix를 다시 다뤄야 하고
- 거대한 matrix에 SVD를 수행하는 것 또한 계산 비용이 크다

⇒ 그래서 Word2Vec의 장점 + count 기반 방식의 장점을 같이 활용하자는 아이디어가 도출된다

### Glove

= Global Vectors for Word Representation

`Word2Vec`의 방식 : 주로 각각의 `local window`를 하나씩 보면서 학습

Glove의 방식 : corpus 전체에서 모은 `global co-occurrence statistics`를 활용

즉, ice와 solid가 몇 번 같이 등장했나?, ice와 gas는?, steam과 solid는?, steam과 gas는? 등과 같은 전체 corpus의 통계를 사용하는 것 !!!

**핵심 아이디어 💥**

단순히 둘이 같이 많이 등장하면 비슷하다에서 더 나아가 두 단어가 다른 단어와 얼마나 자주 등장하는지의 상대적인 관계가 더 의미를 가진다고 본다.

예시로 `ice`, `steam`이라는 두 단어가 있고 그 context word가 `solid`, `gas`, `water`, `fashion` 라고 할때

| context | ice와 함께 | steam과 함께 |
| ------- | ---------- | ------------ |
| solid   | 높음       | 낮음         |
| gas     | 낮음       | 높음         |
| water   | 둘 다 높음 | 둘 다 높음   |
| fashion | 둘 다 낮음 | 둘 다 낮음   |

여기서 중요한 건

- `solid`는 `ice`를 구분하는데 유용
- `gas`는 `steam`을 구분하는데 유용
- `water`는 둘 다 관련 있으므로 상대적 차이는 작음
- `fashion`은 둘 다 관련 없으므로 별 정보 없음

⇒ 단순 등장 횟수 자체보다, **두 단어가 context와 맺는 상대적인 관계가 의미를 더 잘 보여준다**는 것에 집중

⇒ 그래서 Glove는 이를 벡터 공간에 담으려고 한다 ~

+) 이름이 `GloVe`인 이유

- `Word2Vec`: 각 window에서 (center, outside)를 보고 학습 → local context 기반
- `GloVe` : corpus 전체에서 단어 i와 단어 j가 몇번 같이 등장했는지라는 통계를 모아 학습 → global co-occurrence 기반

따라서 `Global Vector` ⇒ `GloVe`가 된 것 ~

**간단 흐름 정리**

```
Word2Vec: 주변 단어를 예측하면서 embedding 학습

그런데...

"그냥 전체 corpus에서 단어가 같이 나온 횟수를 이용하면 안 되나?"

↓

Co-occurrence Matrix: 단어들이 얼마나 자주 같이 등장했는지 기록

↓

문제: 너무 크고 sparse함

↓

SVD: 큰 sparse matrix를 작은 dense vector로 압축

↓

하지만 여전히 계산 비용 등의 한계

↓

GloVe: corpus 전체의 co-occurrence 통계를 word vector 학습에 직접 활용
```

## 3. 단어 임베딩 평가 및 단어 다의성 (Word Sense)

### 벡터값이 잘 나왔는지 평가하는 2가지 방법

#### **1. Word Embedding Evaluation**

- `Intrinsic Evaluation` : word embedding 자체의 품질 평가
  - Word Similarity : 사람의 유사도 판단과 embedding의 유사도가 비슷한지
  - Word Analogy : `man : woman = king : queen`과 같은 의미 관계가 vector 연산으로 표현되는지
- `Extrinsic Evaluation` : NER 등 실제 NLP task에 embedding을 적용해 성능을 평가

#### **2. Word Sense**

하나의 단어가 여러 의미를 가질 수 있음

ex. `bank` = 은행 / 강둑

기존 Word2Vec처럼 단어마다 하나의 embedding을 사용하는 경우 여러 의미가 하나의 vector 안에 함께 표현될 수 있다.

→ 여러 sense vector가 가중되어 합쳐진 **superposition**으로 이해 가능

→ Sparse Coding 등을 통해 이러한 의미들을 분리하는 아이디어도 존재

## 4. 신경망 분류기 기초 (Neural Classifiers)

### **Window Classifier**

center word만 사용하지 않고 주변 단어의 word vector까지 concatenate하여 하나의 입력 벡터로 만든 뒤, 해당 단어의 class를 예측한다.

ex. `[museums in] Paris [are amazing]`

→ 다섯 단어의 embedding을 연결하여 `Paris = Location?`을 판단

### **Neural Classifier**

단순 linear classifier는 선형적인 decision boundary만 만들 수 있다는 한계가 있다.

→ Hidden layer와 `Sigmoid / tanh / ReLU` 등의 non-linear activation function을 추가

→ 복잡한 관계를 표현할 수 있다.

### **Non-linearity가 필요한 이유**

activation function 없이 linear layer만 여러 개 쌓으면

![스크린샷 2026-09-29 오후 5.53.36.png](Lecture%202/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-09-29_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.53.36.png)

처럼 결국 하나의 linear transformation으로 합쳐진다.

따라서 neural network가 복잡한 함수를 학습하려면 **non-linear activation이 필요하다.**
