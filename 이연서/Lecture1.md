## 1. 단어를 숫자로

### One-hot Vector

컴퓨터는 `cat`, `dog`, `banana` 등의 자연어를 이해하지 못한다

→ 따라서 숫자로 표현해야 한다. 그중 가장 단순한 방법이 바로 `One-hot vector` 이다

예시로 카테고리가 `[cat, dog, banana, car]`인 경우

⇒ `cat = [1, 0, 0, 0]` | `dog = [0, 1, 0, 0]` | `banana = [0, 0, 1, 0]` | `car = [0, 0, 0, 1]` 로 표현

### One-hot Vector의 문제점

cat과 dog의 내적 값 = 0 / cat과 banana의 내적 값 = 0

따라서 컴퓨터는 각각의 관계에 아무런 차이가 존재하지 않는다.

그러나 인간은 `cat ↔ dog`가 `cat ↔ banana`보다 훨씬 가깝다는 걸 알고 있다

→ 이게 바로 One-hot Vector의 가장 큰 문제점이다.

### 단어의 ‘의미’를 숫자로 표현하는 방법

#### 1. Distributional Semantics

: 단어의 의미는 그 단어 주변에 어떤 단어들이 등장하는지를 보면 알 수 있다

```
I drink coffee every morning.
I drink tea every morning.

She ordered coffee at the cafe.
She ordered tea at the cafe.
```

위의 문장들을 봤을 때 `coffee`와 `tea` 주변에는 계속 `drink`, `morning`, `ordered`, `cafe` 같은 단어들이 등장한다

따라서 우리는 `context(coffee) ≈  context(tea)` → `meaning(coffee) ≈ meaning(tea)` 로 볼 수 있다 ~ 는 것이 바로 분포 의미론

#### 2. Word Embedding

![스크린샷 2026-09-29 오전 12.12.50.png](Lecture%201/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-09-29_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.12.50.png)

이렇게 **비슷한 단어가 비슷한 벡터를 갖도록** 만드는 것

명칭 : `Dense Vector` / `Word Embedding`

중요 : **사람이** “첫번째 컬럼은 동물성, 두번째 컬럼은 감정” 이런 식으로 **직접 의미 지정 X**

## 2. 벡터를 ‘학습’하는 방법

### Word2Vec

`The quick brown fox jumps over the lazy dog`

- center word = `brown`
- window size = 2

> `window size` : center word 기준으로 몇 칸 옆까지를 "주변"이라고 인정할 것인가?

`[The quick] brown [fox jumps] over the lazy dog`

- outside = `The`, `quick` , `fox`, `jumps`

이때 Word2Vec은 `P(outside | center word)` 를 높이는 방향으로 학습한다

⇒ 해당 방식으로 학습하다보면, 비슷한 문맥에 등장하는 단어들이 비슷한 벡터를 갖게 된다

```
The brown fox jumps.
The red fox jumps.
The brown dog jumps.
The red dog jumps.
```

fox와 dog의
주변에 많은 단어들

      →

```
brown
red
jumps
```

Word2Vec 입장 : fox를 center로 넣었을 경우

`P(brown|fox)`, `P(red|fox)`, `P(jumps|fox)` ⇒ 값을 높여야 한다

마찬가지로 dog를 center로 넣었을 경우

`P(brown|dog)`, `P(red|dog)`, `P(jumps|dog)` ⇒ 값을 높여야 한다

fox와 dog가 비슷한 예측을 해야하는 상황에 놓이기에, 모델이 학습하는 과정에서 fox의 벡터값와 dog의 벡터값을 비슷한 방향으로 조정하게 된다

⇒ 즉, 주변 단어를 잘 맞히는 연습을 시켰더니 그 과정에서 단어의 의미가 벡터 위치에 잘 녹아들었다!!!!

- 비유
  ```
  학생들에게 사람의 성격을 직접 외우라고 하지 않고,
  "이 사람은 누구랑 친할까?"
  "어떤 모임에 자주 나갈까?"

  를 계속 맞히게 하는 거야.
  그걸 수천 번 하다 보면 자연스럽게
  "아, 얘랑 얘는 비슷한 사람이구나."

  라는 구조를 배우게 되는 느낌
  ```

### Word2Vec에서 벡터가 2개인 이유

- center word vector : `v_c`
- outside word vector : `u_o`
- 계산하는 것 : `u_o^Tv_c`

ex. center = `brown`, outside = `fox` ⇒ 계산 : `u_{fox}^Tv_{brown}`

둘이 잘 어울리는 단어일수록 해당 값을 크게 만들고자 하는 것이 목표

### 💥 주의 dot product(내적) 값은 확률이 아니다

`u_{fox}^Tv_{brown}` = 4.2 라고 할 때 4.2라는 값은 확률로 볼 수 없다

→ 그래서 `softmax`를 활용한다!

![스크린샷 2026-09-29 오전 12.42.48.png](Lecture%201/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-09-29_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.42.48.png)

의미) center word `c`가 주어졌을 때, vocabulary에 있는 모든 단어와 점수를 비교해서 `o`가 나올 확률을 계산한다

```
# softmax 적용 전
brown → fox : 4
brown → dog : 2
brown → banana : -1
```

```
# softmax 적용 후
fox     0.87
dog     0.12
banana  0.01/
```

### 진짜 등장한 단어의 확률을 높이는 법

실제 문장에서 brown의 주변에 fox가 등장했다고 가정하면

⇒ Word2Vec의 목표 = `P(fox | brown)` 값을 크게 만드는 것 !!

예를 들어 모델이

- `P(fox | brown) = 0.9`라고 예측 → 실제 등장한 단어에 높은 확률을 줌 ⇒ 잘 예측
- `P(fox | brown) = 0.01`이라고 예측 → 실제 등장한 단어에 낮은 확률을 줌 ⇒ 잘못 예측

따라서 가장 단순하게 생각했을 때의 목표 = `max(P(fox|brown))`

⇒ 즉, 실제로 등장한 outside word의 확률을 최대화하는 것

#### 1. Log를 적용

실제 학습에서는 `max(log P(fox|brown))` 와 같이 `log`를 적용한다

이유) `log`는 확률값의 순서를 바꾸지 않기 때문에, `P(fox|brown)`을 크게 만드는 것과 `log P(fox|brown)`을 크게 만드는 것은 같은 목표로 볼 수 있다.

#### 2. Negative Log-Likelihood

머신러닝에서는 보통 loss를 최소화하는 형태로 학습하기 때문에 앞에 ‘-’ 기호를 붙인다.

`max(log P(fox|brown))` → `min(-log P(fox|brown))`

이때 `-log P(fox|brown)` 부분을 Negative Log-Likelihood(NLL)이라고 볼 수 있다

- 실제 단어에 높은 확률을 부여 → `log` 값이 작음 ⇒ **Loss 작음**
  - `P(fox|brown)=0.9` ⇒ `log(0.9)` ≈ 0.105
- 실제 단어에 낮은 확률을 부여 → `log` 값이 큼 ⇒ **Loss 큼**
  - `P(fox|brown)=0.01` ⇒ `log(0.01)` ≈ 4.605

⇒ 즉, 실제로 등장한 단어를 **높은 확률로 예측할수록 Loss가 작아지도록** 만든 것!!!

#### 3. 모든 center-outside 쌍에 대해 Loss 계산

window size가 2 이고, 문장이 `The quick brown fox jumps` 라면

`brown`을 center로 할 때 실제 outside word는 `The`, `quick`, `fox`, `jumps` 이다

따라서 각각에 대해 `-log` 값을 취한 확률 값( `-log P(The|brown)`, `-log P(quick|brown)`, `-log P(fox|brown)`, `-log P(jumps|brown)`)을 계산한다

그리고 brown 뿐만 아니라 문장의 모든 단어를 차례대로 center word로 두고 같은 과정을 반복한다

![스크린샷 2026-09-29 오전 1.18.49.png](Lecture%201/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-09-29_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_1.18.49.png)

- `T`: 전체 단어 개수
- `t`: 지금 보고 있는 center word 위치
- `w_t`: center word
- `m`: window size
- `j`: center에서 몇 칸 떨어져 있는지
- `w_{t+j}`: 실제 주변 단어

식 전체의 의미 : 모든 `center word`를 하나씩 보면서, 그 주변에 실제로 등장한 단어들의 확률을 높이도록 학습하자 ! ! ( `1/T` 는 그냥 전체 `loss`를 평균내는 부분이라고 생각)

→ 즉, 문자의 모든 center word에 대해 실제 주변에 등장한 단어의 확률을 높여서 전체 Loss를 최대한 작게 만드는 것이 목표 !!!

따라서 이 Loss가 작아지도록 `center word vector`와 `outside word vector`를 계속 수정하면서 `Embedding`을 학습한다.

**간단 정리 !!! 💥**

`-log P(text{actual outside word}|text{center word}))`

- 작다 → 모델이 실제 주변 단어를 높은 확률로 줌 → 잘함
- 크다 → 실제 주변 단어 확률을 낮게 줌 → 못함

## Lecture 1 최종 정리 🌀

컴퓨터는 자연어를 그대로 이해하지 못하기 때문에 **단어를 숫자로 표현할 방법**이 필요

`One-hot Vector`의 등장

↓ 단어를 숫자로 표현할 수 있지만 단어 사이의 의미적 관계를 표현하지 못함

`Distributional Semantics 이론`에 집중

↓ 비슷한 문맥에서 사용되는 단어는 비슷한 의미를 가진다

`Word Embedding`의 탄생

↓ 단어를 의미 정보를 담은 `dense vector`로 표현

- `dense vector` : 값이 빽빽하게 들어찬 벡터 (ex. [0.21,-0.54,0.73,0.11,-0.32, …]) → 단순하게 0이 없다 ~ 라고 이해하기 보단, 이 여러 실수 값에 단어의 의미적/문맥적 특징이 분산되어 표현된다고 이해하기!!
- One-hot vs. Word Embedding
  ```
  One-hot
  cat = [1, 0, 0, 0]

  Embedding
  cat = [0.23, -0.71, 0.48, 0.15, ...]
  ```

`Word2Vec`의 등장

↓ 주변 단어를 예측하도록 학습하면서 동시에 Word Embedding을 학습

식 : `u_o^Tv_c`

↓ 두 단어의 `compatibility score` 값 구하기

- `compatibility score` : center word와 outside word가 얼마나 잘 어울리는지를 나타내는 점수
  - `u_{fox}^{T}v_{brown}` = 4.2 / `u_{banana}^{T}v_{brown}` =-1.3
  - brown ↔ fox 잘 어울림 / brown ↔ banana 잘 안어울림
  - 그러나 확률은 아니고 점수이기에 확률 변환이 필요!!

`Softmax`를 통해 확률값으로 변환

↓ `compatibility score`들을 **모든** vocabulary 단어에 대한 확률분포로 변환

`P(o|c)`

↓ 실제로 등장한 `outside word`의 확률을 높이는 것이 목표

`-logP(o|c)`

↓ 실제 주변 단어에 낮은 확률을 부여하면 큰 Loss(벌점)를 받음

`Gradient Descent` 개념 사용(경사 하강법)

- `Gradient Descent` : Loss가 작아지는 방향을 찾고, 그 방향으로 벡터 값을 조금씩 수정하는 최적화 방법
- `Gradient` : Loss가 가장 빠르게 증가하는 방향
- `- Gradient` : Loss가 감소하는 방향

따라서 `Loss`를 통해 → `Gradient` 계산하고 → `-Gradient` 방향으로 벡터 값 업데이트 → `Loss` 감소하는 흐름

↓ Loss가 작아지는 방향으로 `center` / `outside word vector` 값을 조금씩 수정

이 과정을 수많은 `center-outside word` 쌍에 대해 반복

↓

비슷한 context에서 등장하는 단어들이 자연스럽게 비슷한 vector를 가지게 됨

↓

결과적으로 **단어의 의미적 관계가 Word Embedding 공간에 표현되게 된다 ~ ~**
