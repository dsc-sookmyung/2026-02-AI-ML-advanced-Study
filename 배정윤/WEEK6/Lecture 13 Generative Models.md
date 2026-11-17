# Lecture 13: Generative Models (Part 1)

## 1. 생성 모델이란?

= 데이터 자체의 분포 $p(x)$를 학습. 텍스트 같은 조건 $y$에 맞는 이미지 생성의 경우 조건부 생성 모델 $p(x\mid y)$

지금까지의 분류 모델은 입력 이미지 $x$가 주어졌을 때 레이블 $y$의 조건부 확률 $p(y\mid x)$를 학습.

- **Discriminative:** 이미지 → 분류 결과.
- **Generative:** 데이터 분포 학습 → 새로운 이미지 샘플 생성.
- **Conditional Generative:** 텍스트/클래스 등의 조건 → 그 조건을 따르는 이미지.

생성 모델이 필요한 이유는 같은 조건에서 가능한 결과가 여러 개인 모호성을 하나의 정답이 아니라 확률분포로 표현하기 위해서.

## 2. 생성 모델의 분류

| 분류 | 핵심 특징 | 대표 모델 |
| --- | --- | --- |
| Explicit / tractable density | $p(x)$를 직접 계산 | Autoregressive |
| Explicit / approximate density | $p(x)$를 직접 구하는 대신 근사하거나 하한 사용 | VAE |
| Implicit / direct sampling | 명시적 밀도 계산 없이 직접 샘플 생성 | GAN |
| Implicit / indirect sampling | 반복 절차를 거쳐 샘플 생성 | Diffusion |
|  |  |  |

## 3. 최대가능도학습(Maximum Likelihood)

= 모델이 학습 데이터에 높은 확률을 할당하도록 파라미터를 조정하는 방법.

확률분포를 학습하면 기존 데이터에 얼마나 잘 맞는지를 평가하고, 분포에서 새로운 샘플을 얻는 것이 가능. 

생성 모델 계열마다 이 확률을 표현하거나 학습하는 방식이 상이.

## 4. Autoregressive Models

= 데이터를 순서가 있는 요소의 시퀀스로 보고, **이전 요소들에 조건부로 다음 요소를 예측**.

= 확률의 chain rule이고 결합확률을 조건부 확률의 곱으로 표현.

- **언어:** 앞서 생성한 토큰을 바탕으로 다음 토큰을 예측. LLM도 대표적인 autoregressive model
- **이미지:** 이미지를 픽셀, 서브픽셀 값의 시퀀스로 펼쳐 8비트 서브픽셀마다 0~255 중 하나를 예측.
- **구현:** RNN or masked Transformer로 앞부분 정보만 사용해 순차 예측 가능.

**(+):** 확률을 명시적으로 계산하고 maximum likelihood로 학습 가능.

**(-):** 생성 과정이 순차적이라 고해상도 이미지에서 매우 느림 ex) 1024×1024 RGB 이미지는 약 300만 개의 서브픽셀을 순서대로 다뤄야 함.

## 5. Autoencoder에서 VAE

일반적인 Autoencoder는 입력 $x$를 encoder로 압축해 잠재표현 $z$를 만든 후 decoder로 복원.

- Encoder: $x \rightarrow z$
- Decoder: $z \rightarrow \hat{x}$
- 목표: 입력과 복원 이미지가 최대한 비슷하도록 학습

일반적인 Autoencoder는 잠재공간에서 어디를 샘플링해야 의미 있는 새 이미지를 만들 수 있는지 명시적으로 정의 X. 이를 해결하기 위한 확률적 접근이 VAㄸ

## 6. Variational Autoencoder

VAE는 관측되지 않는 잠재변수 $z$가 먼저 생성, 그 $z$로부터 이미지 $x$가 만들어진다고 가정.

- **Prior:** $p(z)$ — 흔히 표준정규분포 $\mathcal N(0,I)$.
- **Encoder:** $q_\phi(z\mid x)$ — 이미지가 주어졌을 때 잠재변수의 분포를 추정.
- **Decoder:** $p_\theta(x\mid z)$ — 잠재변수에서 이미지의 분포를 모델링.

원하는 $\log p_\theta(x)$를 직접 최적화하기 어려워서 계산 가능한 **ELBO(Evidence Lower Bound)**를 최대화.

**첫 번째 항 — Reconstruction:** 디코더가 입력 이미지를 잘 복원하도록 유도.

**두 번째 항 — KL Divergence:** 인코더가 만드는 잠재분포가 사전분포 $p(z)$에서 크게 벗어나지 않게 한다. 두 항 사이에는 복원 품질과 잠재공간 정규화의 균형이 있다.

### Reparameterization Trick

분포에서 직접 샘플링하는 연산을 그대로 두면 역전파가 어렵다. 따라서 독립적인 표준정규 잡음 $\epsilon$을 뽑아 잠재변수를 만든다. 무작위성은 $\epsilon$에 두면서 encoder의 $\mu,\sigma$까지 gradient를 전달할 수 있다.

### VAE로 새로운 이미지 생성

1. $z\sim\mathcal N(0,I)$를 샘플링한다.
2. 학습된 decoder에 $z$를 입력한다.
3. decoder 출력으로 새 이미지를 생성한다.

**(-):** VAE 이미지가 흐릿해질 수 있으며, reconstruction과 latent regularization 사이에 trade-off가 존재한다.