# Lecture 3 - Backpropagation, Neural Network

# 1. 신경망 학습: 기울기 계산과 역전파

## 신경망의 구조 (Matrix Notation)

신경망은 여러 개의 로지스틱 회귀를 동시에 실행하는 것과 같다. 층을 쌓으면 데이터를 여러 번 재표현·조합해서 원래 입력에 대해 매우 비선형적인 분류기를 학습할 수 있다.

```
z = Wx + b
a = f(z)      (f는 element-wise로 적용)
```

## 비선형 함수 (Non-linearity)

**왜 필요한가**

- 비선형이 없으면 여러 층이 `W₁W₂x = Wx`처럼 하나의 선형 변환으로 합쳐져 버린다.
- 비선형이 있어야 층을 쌓아 복잡한 함수를 근사할 수 있다.

## 기울기 (Gradient)와 Jacobian

- 입력 1개, 출력 1개: 미분 = 기울기 (예: x³의 미분은 3x²)
- 입력 n개, 출력 1개: gradient는 각 입력에 대한 편미분 벡터
- 입력 n개, 출력 m개: **Jacobian**은 m × n 행렬이고, (i, j) 성분은 ∂fᵢ/∂xⱼ

## Chain Rule

- 단일 변수 함수의 합성: 미분을 곱한다.
- 다변수 함수의 합성: **Jacobian을 곱한다.**

```
h = f(z), z = Wx + b
∂h/∂x = (∂h/∂z)(∂z/∂x)
```

## 계산 그래프와 역전파 (Backpropagation)

역전파는 chain rule을 계산 그래프에서 재귀적으로 적용하되, 상위 층에서 계산한 미분을 하위 층에서 재사용해 계산량을 줄이는 것이다.

- **계산 그래프**: source node는 입력, interior node는 연산, edge는 연산 결과를 전달
- **Forward propagation**: 그래프를 따라 값을 계산하고 중간값을 저장
- **Backward propagation**: 그래프를 거꾸로 따라가며 gradient를 전달

```
[downstream gradient] = [upstream gradient] × [local gradient]
```

- upstream gradient: 출력 쪽에서 들어오는 gradient
- local gradient: 노드의 출력을 입력으로 미분한 값
- downstream gradient: 입력 쪽으로 전달하는 gradient
- 입력이 여러 개인 노드는 입력마다 local gradient가 있다.

## 자동 미분과 구현

- 딥러닝 프레임워크(TensorFlow, PyTorch 등)는 fprop의 수식으로부터 gradient 계산을 자동으로 추론한다.
- 각 노드(gate)는 forward(출력 계산)와 backward(출력 gradient를 받아 입력 gradient 계산)를 구현한다.
- 예: MultiplyGate는 forward에서 x, y를 저장해 두고(`self.x`, `self.y`), backward에서 dx = y·dz, dy = x·dz를 반환한다.
- 프레임워크가 backprop을 해주지만, 각 층의 local derivative는 층 작성자가 직접 계산해야 한다.

## Numeric Gradient (Gradient Checking)

```
f'(x) ≈ (f(x + h) − f(x − h)) / 2h     (h ≈ 1e-4)
```

- 구현이 쉽고 정확하게 만들기 쉽지만, 근사값이고 파라미터마다 f를 다시 계산해야 해서 매우 느리다.
- 직접 구현한 layer의 gradient가 맞는지 검증하는 용도로 유용하다.

## 왜 역전파를 직접 이해해야 하는가

- 프레임워크가 대신 계산해 주지만, 내부 동작을 이해해야 디버깅과 모델 개선이 가능하다.
- Backprop은 그대로는 항상 잘 동작하지 않는다. (예: exploding / vanishing gradient는 이후 강의에서 다룸)
- 참고: Karpathy의 "Yes you should understand backprop"