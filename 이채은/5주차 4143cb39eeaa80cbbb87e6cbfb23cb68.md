# 5주차

# CS231N Lecture 12: Self-Supervised Learning (SSL)

## 1. Self-Supervised Learning (SSL) 개요

- **배경 및 문제점:** 기존의 지도 학습(Supervised Learning)은 성능이 뛰어나지만, 정답(Label)이 달린 방대한 데이터셋이 필수적입니다. 데이터 라벨링은 많은 비용과 시간을 요구하는 병목 현상을 유발합니다.
- **SSL의 핵심 아이디어:** 라벨이 없는 원시 데이터(Raw Unlabeled Data) 자체에서 정답을 만들어내어 모델을 학습시키는 방법입니다.
- **학습 파이프라인:**
    1. **Pretext Task (사전 학습 과제):** 라벨 없이 데이터의 내재적 구조를 활용해 모델이 스스로 문제를 만들고 푸는 과정입니다. 이를 통해 모델은 범용적인 시각적 특징(Feature Representation)을 학습합니다.
    2. **Downstream Task (하위/실제 과제):** 사전 학습된 모델(Encoder)을 가져와, 소량의 라벨링 데이터만으로 목표하는 실제 문제(분류, 객체 탐지, 세그멘테이션 등)에 맞게 전이 학습(Transfer Learning)을 진행합니다.

## 2. 초기 Pretext Tasks (휴리스틱 기반 방법론)

모델이 이미지의 구조와 문맥(Context)을 이해하도록 설계된 초기 기법들입니다.

- **Predicting Relative Position (상대적 위치 예측):** 이미지에서 두 개의 패치를 추출하고, 한 패치가 다른 패치 기준 어느 방향에 있는지 예측합니다.
- **Jigsaw Puzzles (직소 퍼즐):** 이미지를 9개의 패치로 나누어 섞은 뒤, 원래의 패치 순서(미리 정의된 순열 중 하나)를 예측하여 이미지의 공간적 배치를 학습합니다.
- **Rotation (회전 각도 예측):** 이미지를 0도, 90도, 180도, 270도 중 하나로 회전시키고, 회전된 각도를 4-way 분류 문제로 풉니다. 모델이 객체의 정방향 형태를 인지해야만 풀 수 있습니다.
- **Image Inpainting (이미지 복원):** 이미지의 특정 영역을 마스킹하여 지우고, 주변 픽셀과 문맥을 바탕으로 지워진 영역을 예측해 채워 넣습니다.
- **Image Colorization (색상 복원):** 컬러 이미지를 밝기 채널(L)과 색상 채널(ab)로 분리한 뒤, 밝기(흑백) 정보만 입력받아 원래의 색상 정보를 예측하도록 학습합니다.

## 3. Contrastive Learning (대조 학습)

초기 Pretext Task들의 한계(휴리스틱 의존성)를 극복하기 위해 등장한 기법으로, 데이터 간의 유사성을 바탕으로 특징 공간(Latent Space)을 최적화합니다.

- **핵심 원리:**
    - **Positive Pairs:** 동일한 이미지에 서로 다른 데이터 증강(Crop, Color Jitter 등)을 적용한 두 이미지. 이들의 특징 벡터 거리는 가깝게(Attract) 당깁니다.
    - **Negative Pairs:** 배치(Batch) 내에 포함된 완전히 다른 이미지들. 이들의 특징 벡터 거리는 서로 멀어지게(Repel) 밀어냅니다.
- **InfoNCE Loss:** Positive 샘플과의 유사도는 최대화하고, 다수의 Negative 샘플들과의 유사도는 최소화하도록 설계된 대조 손실 함수입니다.

### 3.1. SimCLR (2020)

- **구조:** [데이터 증강 → Base Encoder (ResNet) → Projection Head (MLP) → Contrastive Loss]의 직관적인 파이프라인을 가집니다.
- **한계점:** 대조 학습의 특성상 수많은 Negative 샘플과 비교해야 높은 성능을 낼 수 있으므로, 매우 큰 Batch Size(예: 4096 이상)와 막대한 메모리가 필수적으로 요구됩니다.

### 3.2. MoCo (Momentum Contrast, 2020)

- **목적:** SimCLR의 거대한 Batch Size 요구 문제를 해결하기 위해 고안되었습니다.
- **Memory Queue (메모리 큐):** 과거 배치에서 연산된 Negative 샘플의 특징(Feature)들을 버리지 않고 Queue 형태로 저장하여 유지합니다. 이를 통해 Batch Size를 키우지 않아도 방대한 Negative 샘플을 확보할 수 있습니다.
- **Momentum Encoder:** Queue에 저장된 특징 벡터들의 일관성을 유지하기 위해, Key(Negative) 인코더는 역전파(Backpropagation)로 학습시키지 않고 Query 인코더의 가중치를 이동 평균(Moving Average) 방식으로 천천히 업데이트합니다.

### 3.3. BYOL & SimSiam

- Negative 샘플 없이 Positive Pair만으로 학습을 진행하는 모델들입니다.
- 일반적으로 Negative 샘플이 없으면 모델의 출력이 하나의 값으로 수렴하는 붕괴 현상(Collapse)이 발생하지만, 이 모델들은 Predictor 네트워크 추가 및 Stop-gradient(그래디언트 차단) 연산을 도입한 비대칭 구조를 통해 붕괴 현상을 방지합니다.

## 4. Masked Image Modeling (MIM)

자연어 처리(NLP)의 BERT(Masked Language Modeling) 방법론을 컴퓨터 비전 분야에 성공적으로 적용한 최신 주류 방법론으로, 주로 Vision Transformer(ViT) 구조를 사용합니다.

### 4.1. MAE (Masked Autoencoders, 2021)

- **핵심 원리:** 입력 이미지 패치 중 매우 높은 비율(약 75%)을 가리고(Masking), 남은 25%의 정보만으로 전체 이미지를 완벽히 복원(Reconstruction)하는 난도 높은 학습을 수행합니다.
- **비대칭(Asymmetric) 구조의 이점:**
    - **Encoder:** 가려진 패치를 과감히 버리고, 오직 **보이는(Visible) 25%의 패치만** 연산합니다. 이로 인해 메모리 사용량과 연산 속도가 극적으로 개선되어 거대 비전 모델 학습이 가능해집니다.
    - **Decoder:** Encoder가 추출한 특징에 학습 가능한 'Mask Token'을 결합하여 원본을 픽셀 단위로 복원합니다. 사전 학습이 끝나면 Decoder는 폐기하고 Encoder만 Downstream Task에 활용합니다.
- **효과:** 단순한 패턴 매칭이 불가능할 정도로 많이 마스킹하기 때문에, 모델이 객체의 전체적인 구조와 의미(Semantic)를 깊이 있게 파악하도록 강제합니다.

## 5. 모델 평가 방법 (Evaluation)

학습된 Self-Supervised 모델이 얼마나 좋은 특징을 추출하는지 검증하기 위해 두 가지 방식을 활용합니다.

1. **Linear Probing (선형 평가):** 학습된 Encoder의 가중치를 고정(Freeze)하고 업데이트하지 않습니다. 그 위에 단순한 선형 분류기(Linear Classifier) 계층 하나만 추가하여 학습시킨 뒤 성능을 측정합니다. 모델이 추출한 특징이 얼마나 선형적으로 잘 분리되는지 엄밀하게 평가할 때 사용합니다.
2. **Fine-tuning (미세 조정):** Downstream Task의 데이터로 Encoder와 분류기의 가중치를 모두 함께 업데이트합니다. 실제 서비스나 애플리케이션에 모델을 적용할 때 가장 일반적으로 사용하는 방식입니다.

# CS231N Lecture 13: Generative Models 1 (생성 모델 개요 및 Auto-regressive & VAE)

## 1. 생성 모델(Generative Models) 개요

- **분류 모델(Discriminative Model)과의 차이점:**
    - **Discriminative Model:** 입력 데이터(X)가 주어졌을 때 정답 레이블(Y)의 조건부 확률 $P(Y|X)$를 학습합니다. 이미지들이 확률 질량(Probability mass)을 두고 경쟁하지 않으며 고정된 카테고리 내에서 확률을 나눕니다.
    - **Generative Model:** 정답(Y) 없이 데이터(X) 자체의 분포인 $P(X)$를 학습합니다. 세상의 모든 이미지가 확률 질량을 두고 경쟁해야 하므로, 생성 모델은 현실에 존재할 법한 이미지가 무엇인지 깊은 수준의 이해를 요구합니다.
- **사용 목적:** 입력에 대한 출력값이 하나로 결정되지 않는 모호성(Ambiguity)을 다룰 때 필수적입니다. (예: "생성 모델에 대한 시를 써줘", "고양이가 있는 사진을 그려줘"라는 요청에는 무한히 많은 정답이 존재함)

## 2. 생성 모델의 분류 (Taxonomy)

생성 모델은 확률 밀도 함수(Density Function)를 어떻게 다루느냐에 따라 크게 두 갈래로 나뉩니다.

- **Explicit Density (명시적 밀도 모델):** 데이터의 확률 밀도 $P(X)$ 값을 직접 계산할 수 있는 모델입니다.
    - *Tractable (정확한 계산):* Auto-regressive Models (정확한 $P(X)$ 도출 가능)
    - *Approximate (근사치 계산):* Variational Autoencoders (VAE) (하한선으로 근사)
- **Implicit Density (암시적 밀도 모델):** 확률 밀도 $P(X)$ 값을 계산할 수는 없으나, 그 분포로부터 샘플링하여 새로운 데이터를 생성할 수 있는 모델입니다.
    - *Direct (직접 샘플링):* GAN (Generative Adversarial Networks)
    - *Indirect (반복적 샘플링):* Diffusion Models (확산 모델)

## 3. Auto-regressive Models (자기회귀 모델)

명시적 밀도 모델 중 정확한 확률 $P(X)$를 구할 수 있는 방법론입니다.

- **핵심 원리:** 체인 룰(Chain Rule of Probability)을 이용해 다차원 데이터의 결합 확률을 1차원 조건부 확률의 곱으로 분해합니다.
    - $P(X) = P(X_1) \times P(X_2|X_1) \times P(X_3|X_1, X_2) \dots$
- **적용:**
    - 언어 모델(RNN, Transformer 등)에 매우 자연스럽게 적용됩니다. 텍스트는 본질적으로 1차원 순차(Sequence) 데이터이기 때문입니다.
    - 이미지에 적용할 경우(PixelRNN 등), 이미지를 1차원 픽셀 나열로 펼쳐서 이전 픽셀들을 바탕으로 다음 픽셀을 예측하게 만듭니다.
- **장단점:** 확률 밀도를 완벽하게(Tractable) 계산할 수 있다는 장점이 있지만, 고해상도 이미지(예: 1024x1024)를 1차원 픽셀로 다루면 시퀀스가 수백만 개로 늘어나 연산 비용이 기하급수적으로 증가(매우 비효율적)하는 치명적인 단점이 있습니다.

## 4. Variational Autoencoders (VAE)

정확한 확률 밀도 계산을 포기하는 대신, 의미 있는 **Latent Vector (잠재 벡터, $Z$)**를 얻는 것을 목표로 하는 모델입니다.

- **배경 개념 (Autoencoder):** 입력(X)을 저차원 잠재 공간(Z)으로 압축하는 Encoder와, 이를 다시 원래 입력(X)으로 복원하는 Decoder로 구성됩니다. 하지만 일반적인 Autoencoder의 $Z$ 공간은 연속성이나 확률적 규칙이 없어 새로운 이미지를 생성(Sampling)하기 어렵습니다.
- **VAE의 등장:** $Z$ 공간이 우리가 알고 있는 분포(주로 표준 정규분포, Standard Gaussian)를 따르도록 강제합니다. 학습이 끝나면 정규분포에서 무작위 노이즈 $Z$를 샘플링한 뒤 Decoder에 넣어 새로운 이미지를 생성할 수 있습니다.
- **수식과 ELBO (Evidence Lower Bound):**
    - VAE는 데이터의 $P(X)$를 직접 최대화할 수 없으므로(적분이 불가능함), 수학적인 식 조작(베이즈 정리 및 KL-Divergence 활용)을 통해 실제 로그 우도(Log-likelihood)보다 작거나 같은 하한선인 ELBO를 유도합니다.
    - ELBO를 최대화하면 간접적으로 모델의 우도가 최대화됩니다.
- **VAE Loss의 두 가지 축 (경합):**
    1. **Reconstruction Loss (복원 오차):** 원본 데이터 $X$를 잃어버리지 않고 똑같이 복원하려는 성질. (MSE 형태와 유사)
    2. **KL-Divergence (사전 분포와의 차이):** 인코더가 예측한 $Z$ 분포가 지정해 둔 정규분포(Prior) 형태를 따르도록 강제하려는 성질.
    - 두 Loss가 싸우면서 균형을 찾아가며, 결과적으로 부드럽게 연속적으로 변하는(Interpolation이 가능한) 의미 있는 잠재 공간(Latent Space)을 학습합니다.

# CS231N Lecture 14: Generative Models 2 (GAN, Diffusion, 그리고 Modern Pipeline)

## 1. Generative Adversarial Networks (GAN)

명시적인 밀도 함수 $P(X)$를 전혀 구하지 않고(Implicit), 게임 이론을 활용하여 데이터 분포와 똑같은 샘플을 만들어내는 것에만 집중하는 모델입니다.

- **핵심 구조 (Minimax Game):**
    - **Generator (생성자, G):** 무작위 노이즈 $Z$를 입력받아 가짜 이미지 $X$를 만들어냅니다. Discriminator를 속이는 것이 목표입니다.
    - **Discriminator (판별자, D):** 들어온 이미지가 진짜(Real) 훈련 데이터인지 생성자가 만든 가짜(Fake)인지 판별하는 이진 분류기입니다.
- **학습 원리:** 두 네트워크가 서로 경쟁(Adversarial)합니다. Discriminator는 진짜와 가짜를 완벽히 구별하도록 학습하고, Generator는 판별자가 가짜 이미지를 진짜로 착각할 정도로 실제와 똑같은 데이터를 만들도록 학습합니다.
- **장단점:**
    - **장점:** 매우 선명(Crisp)하고 고해상도의 뛰어난 이미지를 생성할 수 있습니다. (VAE 특유의 흐릿함 극복)
    - **단점 (치명적):** 명확한 Loss 지표가 없어(판별자 성능이 변하기 때문) 학습 안정성이 극도로 떨어집니다. Mode Collapse(생성자가 몇 가지 데이터만 기억해서 반복 출력하는 현상)가 발생하기 쉽고 하이퍼파라미터 튜닝이 매우 어렵습니다.

## 2. Diffusion Models (확산 모델)

최근 수년간 GAN을 밀어내고 이미지/비디오 생성의 최고 존엄으로 자리 잡은 방법론입니다. 복잡한 문제를 한 번에 풀지 않고, 노이즈를 아주 조금씩 제거해 나가는 반복적(Iterative) 접근 방식을 취합니다.

- **학습 과정 (Forward / Training):**
    - 깨끗한 데이터(이미지)에 점진적으로 노이즈(Gaussian)를 추가하여 완전히 알아볼 수 없는 노이즈 상태로 만듭니다. (시간 척도 $t=0$ 은 깨끗함, $t=1$ 은 완전한 노이즈)
    - 신경망(주로 UNet이나 최근엔 DiT)은 특정 노이즈 단계($t$)의 이미지에 들어간 '노이즈 성분' 혹은 '원래 이미지 방향(Velocity)'을 예측하도록 학습합니다 (단순 MSE Loss 사용).
- **생성 과정 (Inference / Reverse):**
    - 완전한 무작위 노이즈 분포에서 출발하여, 학습된 신경망을 수십~수백 번 반복 적용하면서 노이즈를 아주 조금씩 걷어냅니다.
    - 최종적으로 깨끗한 생성 이미지를 얻습니다.
- **Rectified Flow 기반의 직관적 해석:** 데이터 점과 노이즈 점 사이에 선형적인 선을 긋고, 이 선을 따라 이동하는 속도 벡터(Velocity Vector)를 예측하게 학습시킵니다. 생성 시에는 노이즈에서 출발하여 이 속도 벡터를 따라 여러 번의 Step을 거쳐 데이터로 도달하는 것과 같습니다.
- **장단점:**
    - **장점:** GAN과 달리 Loss 곡선이 명확하여 학습이 훨씬 안정적이고 고품질 샘플링이 가능합니다.
    - **단점:** 생성 시 네트워크를 여러 번 거쳐야 하므로 계산 속도가 굉장히 느립니다. (이를 해결하기 위해 샘플링 스텝을 대폭 줄이는 'Distillation' 연구가 활발합니다.)

## 3. 현대적 생성 모델 파이프라인 (Latent Diffusion Models)

고해상도 이미지나 수만 토큰에 달하는 고프레임 비디오에 픽셀 단위로 Diffusion 연산을 적용하면 너무 무겁습니다. 이를 해결하기 위해 현업에서는 앞서 배운 모든 생성 모델 기술(VAE, GAN, Diffusion)을 결합하여 사용합니다.

- **Latent Diffusion 파이프라인 구조:**
    1. **VAE 기반 Encoder:** 먼저 거대한 원본 이미지(혹은 비디오)를 차원이 축소된 압축된 잠재 공간(Latent Space) 벡터로 변환시킵니다. (연산량 극적 감소)
    2. **Diffusion Model 적용:** 픽셀이 아닌 이 가벼운 '잠재 공간' 위에서 노이즈를 더하고 빼는 Diffusion 학습 및 생성을 수행합니다.
    3. **VAE 기반 Decoder (+ GAN Loss):** 노이즈를 걷어내 완성된 깨끗한 잠재 벡터를 다시 디코더를 통해 실제 픽셀 이미지로 복원합니다. 이때 VAE의 흐릿함을 없애고 선명한 결과를 내기 위해 디코더 학습 시 GAN의 Discriminator를 함께 도입하여 고품질을 확보합니다.
- **Conditioning (조건 제어):** 텍스트 프롬프트(Text Prompt) 등 사용자의 입력을 반영하기 위해 CLIP이나 T5 같은 언어 모델이 뽑아낸 텍스트 벡터를 Diffusion Model 중간(Cross-attention 등)에 주입합니다.
- **CFG (Classifier-Free Guidance):** 생성 시 프롬프트를 무시하고 자기 마음대로 생성하는 것을 막기 위해, 프롬프트를 줬을 때의 예측 벡터와 프롬프트를 주지 않았을 때(Null) 예측 벡터의 차이를 과장시켜서(가중치를 주어) 프롬프트 반영률을 폭발적으로 끌어올리는 필수 테크닉입니다.