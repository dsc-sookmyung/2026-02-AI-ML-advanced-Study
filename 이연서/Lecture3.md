# Lecture 3

개념: Backpropagation and Neural Network Basics
과제: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/assignments/a1_preview/exploring_word_vectors.html
과제 제출일: 2026년 9월 29일
날짜: 2026년 9월 29일

## 1. Backpropagation이 필요한 이유

신경망에는 `Weight`, `Bias`, `Embedding` 등 많은 parameter가 존재한다.

Loss를 줄이기 위해서는 각 parameter가 Loss에 얼마나 영향을 미치는지 알아야 한다.

→ 이를 계산하는 방법이 `Backpropagation`

### Backpropagation vs Gradient Descent

- `Backpropagation` : 각 parameter에 대한 Gradient를 계산
- `Gradient Descent` : 계산된 Gradient를 이용해 parameter를 업데이트

전체 흐름

`Forward Pass` → `Loss 계산` → `Backpropagation으로 Gradient 계산` → `Gradient Descent로 parameter 업데이트`

## 2. 미분과 Chain Rule

### 미분의 의미

미분은 쉽게 말하면

> 입력값이 조금 변했을 때 출력값이 얼마나 변하는지
> 

를 나타낸다.

신경망에서는 `dJ/dW`를 통해

> Weight `W`가 조금 변했을 때 Loss `J`가 얼마나 변하는지
> 

를 계산한다.

### Chain Rule

신경망은 여러 연산이 연속해서 연결되어 있기 때문에 중간 연산을 모두 고려해야 한다.

ex.

`W → z → h → Loss`

이때 `W`가 Loss에 미치는 영향은 각 단계의 변화율을 곱해서 계산한다.

즉, **Loss의 W에 대한 Gradient =** `Loss의 h에 대한 Gradient` X `h의 z에 대한 Gradient` X `z의 W에 대한 Gradient`

⇒ 이게 바로 `Chain Rule`

## 3. Forward Pass / Backward Pass

### Forward Pass

입력값부터 시작해 각 layer의 값을 순서대로 계산하는 과정

`Input` → `중간 연산`→ `Output`→ `Loss`

Backward Pass에서 사용하기 위해 중간 계산값도 함께 저장해야 한다

### Backward Pass

Loss에서 시작해 역방향으로 Gradient를 계산하는 과정

→ 각 값 또는 parameter가 최종 Loss에 얼마나 영향을 미치는지 계산

핵심 관계

`Downstream Gradient` = `Upstream Gradient`×`Local Gradient`

- `Upstream Gradient` : 뒤쪽에서 전달된 Gradient
- `Local Gradient` : 현재 연산 자체의 Gradient
- `Downstream Gradient` : 이전 노드로 전달할 Gradient

## 4. Backpropagation 핵심

Backpropagation은 `Chain Rule`을 이용해 출력층에서 입력 방향으로 Gradient를 전달한다.

- `Forward Pass` → 실제 값과 Loss 계산
- `Backward Pass`→ 각 parameter에 대한 Gradient 계산
- `Gradient Descent` → Gradient의 반대 방향으로 parameter 업데이트

즉,

> `Backpropagation = Gradient 계산`
> 
> 
> `Gradient Descent = 계산된 Gradient로 parameter 수정`
> 

## 5. Activation Function

신경망에서 여러 Linear Layer만 연속으로 쌓으면 결국 하나의 Linear Transformation으로 합쳐진다.

예) `W2(W1x) = Wx`

→ 따라서 복잡한 패턴을 학습하려면 중간에 **Non-linear Activation Function**이 필요

### 주요 Activation Function

- `Sigmoid`
    - 출력 범위 : 0 ~ 1
    - 과거에 많이 사용
    - 입력값의 절댓값이 커지면 Gradient가 매우 작아질 수 있음
- `tanh`
    - 출력 범위 : -1 ~ 1
    - Sigmoid와 비슷하지만 0을 중심으로 값을 가짐
- `ReLU`
    - 입력이 0보다 크면 그대로 출력
    - 입력이 0보다 작으면 0 출력
    - 계산이 단순하고 Gradient 전달이 잘 되어 널리 사용
- `Leaky ReLU`
    - ReLU와 달리 음수 영역에서도 작은 값을 허용
    - Dead ReLU 문제를 완화
- `GELU / Swish`
    - 최근 신경망 및 Transformer 계열에서 자주 사용하는 Activation Function

## 6. Matrix Calculus

신경망에서는 값 하나가 아니라 Vector와 Matrix를 이용하기 때문에 미분도 Vector / Matrix 단위로 계산해야 함

기본 원리는 일반 미분과 동일

> 입력값이 변할 때 출력값이 얼마나 변하는지를 계산
> 

예) `dJ/dW`→ Matrix `W`를 변화시켰을 때 Loss `J`가 어떻게 변하는지 나타낸다.

### Gradient Shape

Parameter를 Gradient Descent로 업데이트하려면 `Parameter의 Shape = Gradient의 Shape`이어야 한다.

예) `W.shape = (3, 4)` 라면 `dJ/dW.shape = (3, 4)` 이어야 한다.

→ 그래야 `W - learning_rate × gradient` 형태로 업데이트할 수 있다.

## 7. Jacobian Matrix

입력과 출력이 모두 여러 개의 값으로 이루어진 Vector라면, 하나의 미분값만으로는 관계를 표현할 수 없다.

예) 입력 `x = [x1, x2]`/ 출력 `y = [y1, y2, y3]`

이라면 각각

- x1이 y1에 미치는 영향
- x2가 y1에 미치는 영향
- x1이 y2에 미치는 영향
- x2가 y2에 미치는 영향
- ...

을 모두 계산해야 한다.

이 값들을 하나의 Matrix로 정리한 것이 `Jacobian Matrix` 입력이 `n개`, 출력이 `m개`라면 `Jacobian Shape = (m, n)`

> 즉, Jacobian = 여러 입력과 여러 출력 사이의 모든 미분 관계를 모아놓은 Matrix
> 

## 8. Element-wise Function의 Jacobian

Activation Function은 보통 Vector의 각 원소에 독립적으로 적용된다.

예) `x = [x1, x2, x3]`

→ `f(x) = [f(x1), f(x2), f(x3)]`

이때

- x1은 f(x1)에만 영향
- x2는 f(x2)에만 영향
- x3는 f(x3)에만 영향

을 주기 때문에 서로 다른 원소 사이의 미분값은 0이 된다.

따라서 Jacobian은 **Diagonal Matrix** 형태가 된다.

```
[f'(x1)    0       0   ]
[   0    f'(x2)    0   ]
[   0       0    f'(x3)]
```

→ Element-wise Activation Function의 미분 계산이 비교적 단순한 이유

## 9. 역전파 알고리즘 (Backpropagation)

`Backpropagation`은 최종 Loss에서 시작해 역방향으로 이동하면서 각 parameter의 Gradient를 계산하는 알고리즘이다. ⇒ 핵심 원리는 `Chain Rule` !!!

### Forward Pass

입력부터 출력 방향으로 계산을 진행한다.

`Input` → `Hidden Layer` → `Output` → `Loss`

이 과정에서 **Backward Pass에 필요한 중간 계산값들을 저장**한다.

### Backward Pass

Loss에서 입력 방향으로 돌아가면서 각 값이 최종 Loss에 얼마나 영향을 미쳤는지 계산한다.

핵심 관계: `Downstream Gradient`

`Upstream Gradient × Local Gradient`

- `Upstream Gradient` : 뒤쪽에서 전달받은 Gradient
- `Local Gradient` : 현재 연산 자체의 Gradient
- `Downstream Gradient` : 이전 노드로 전달할 Gradient

> 즉, 뒤에서 받은 Gradient에 현재 연산의 미분값을 곱하면서 역방향으로 전달한다.
> 

### Backpropagation의 핵심

`Forward Pass` → 값과 Loss 계산 + 중간값 저장

`Backward Pass` → Chain Rule을 이용하여 Gradient 계산

`Gradient Descent` → 계산한 Gradient를 이용하여 parameter 업데이트

## 10. Computational Graph의 Gradient 전달

복잡한 신경망 연산을 작은 연산 단위의 `Computational Graph`로 표현할 수 있다. 또한 각 연산에 따라 Gradient를 전달하는 방식이 달라진다.

### Plus Gate (+)

: Gradient를 그대로 전달

`x + y` → x와 y 모두 같은 Upstream Gradient를 전달받음

### Multiply Gate (×)

: 상대방의 입력값을 곱해서 Gradient를 전달

`x × y`

- x 방향 Gradient → `Upstream Gradient × y`
- y 방향 Gradient → `Upstream Gradient × x`

### Max Gate

Forward Pass에서 더 큰 값을 선택했던 경로로만 Gradient를 전달

→ 선택되지 않았던 경로의 Gradient는 0

### Branching

하나의 값이 여러 연산에 사용되어 Gradient가 여러 경로를 통해 돌아올 경우

→ **각 경로에서 전달된 Gradient를 모두 더한다.**

## 11. 수치적 기울기 검증 (Gradient Checking)

Backpropagation으로 계산한 Gradient가 제대로 구현되었는지 확인하기 위한 방법

→ 아주 작은 값 `h`만큼 parameter를 직접 변화시킨 뒤 Loss가 얼마나 변하는지 확인한다.

중앙 차분 방식: `Gradient ≈ [f(x + h) - f(x - h)] / 2h`

1. parameter를 아주 조금 증가시킨 결과 계산
2. parameter를 아주 조금 감소시킨 결과 계산
3. 두 값의 차이를 이용해 실제 Gradient를 근사

### 왜 사용하는가?

Backpropagation으로 계산한 Gradient와 수치적으로 계산한 Gradient를 비교

→ 두 값이 거의 같다면 Backpropagation 구현이 올바를 가능성이 높음

단, 모든 parameter에 대해 함수를 여러 번 계산해야 하므로 **매우 느리다.**

따라서 실제 모델 학습에 사용하는 방법이라기보다 Backpropagation 구현이 정확한지 확인하는 검증용 !!!!!!