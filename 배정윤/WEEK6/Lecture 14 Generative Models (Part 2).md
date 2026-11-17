# Lecture 14: Generative Models (Part 2)

## 1. Generative Adversarial Networks — GAN

= **Generator와 Discriminator를 경쟁적으로 학습**해 실제 데이터와 비슷한 이미지를 생성하는 모델.

- **Generator:** Latent variable에서 fake image 생성.
- **Discriminator:** Real / fake 판별.
- **(+):** 선명한 이미지 생성, 빠른 inference.
- **(-):** Training Instability, Mode Collapse.

## 2. GAN Training Objective — Minimax Game

= Discriminator는 real/fake를 정확하게 구별하고, Generator는 Discriminator를 속이도록 학습.

- **Discriminator:** Objective 최대화(Gradient Ascent).
- **Generator:** Objective 최소화(Gradient Descent).
- 두 network를 번갈아 업데이트. Loss 값만으로 생성 품질 판단 어려움.

## 3. GAN Training Challenges & Improvements

- **Vanishing Gradient:** Discriminator가 fake를 너무 쉽게 구별하면 Generator 학습이 어려움.
- **Non-saturating Loss:** Generator가 fake를 real로 판단하게 만드는 방향으로 학습해 gradient 개선.
- **Mode Collapse:** 서로 다른 입력에도 비슷한 이미지만 반복 생성.
- **DCGAN / StyleGAN:** CNN 기반 구조 / latent interpolation을 통한 자연스러운 이미지 변화.

## 4. Diffusion Models

= **Random noise를 반복적으로 제거**하여 이미지를 생성하는 모델.

- **Training:** 실제 이미지에 noise를 추가하고 제거하는 방법 학습.
- **Sampling:** Pure noise → 반복적 denoising → image.
- GAN보다 생성은 느리지만 학습이 상대적으로 안정적.

## 5. Rectified Flow Models

= 실제 data와 noise 사이의 **velocity(이동 방향)**를 예측하는 방법.

- **Training:** Data와 noise를 섞어 만든 sample에서 velocity를 예측하고 MSE 최소화.
- **Sampling:** Pure noise에서 예측한 velocity의 반대 방향으로 반복 이동.
- GAN보다 학습이 안정적인 편이지만 여러 sampling steps 필요.

## 6. Conditional Diffusion & Classifier-Free Guidance

- **Conditional Diffusion:** Text prompt, class label 등 조건에 맞는 이미지 생성.
- **Classifier-Free Guidance(CFG):** Conditional / unconditional prediction을 결합해 조건 반영 강도 조절.
- **(+):** Prompt 반영 강화.
- **(-):** Sampling cost 증가.

## 7. Noise Schedule

= 학습할 **noise level의 분포와 비중**을 결정하는 방법.

- 모든 noise level의 예측 난이도가 동일하지 않음.
- **Logit-normal:** 어려운 중간 noise level에 더 높은 비중.
- **Shifted Schedule:** High-resolution 등 데이터 특성에 맞게 noise level 조절.

## 8. Latent Diffusion Models — LDM

= 원본 pixel이 아닌 **압축된 latent space에서 diffusion**을 수행해 연산량 감소.

- **Encoder / Decoder:** Image 압축 및 복원.
- **Diffusion:** Latent에서 반복적으로 noise 제거.
- **VAE + GAN + Diffusion:** Latent 표현, 선명한 복원, 새 이미지 생성을 결합.

## 9. Diffusion Transformers — DiT

= Diffusion의 denoising network에 **Transformer**를 사용하는 구조.

- **입력:** Noisy latent, timestep, text 등의 condition.
- **Scale & Shift:** Timestep 반영.
- **Cross / Joint Attention:** Text 등 조건 정보 결합.

## 10. Text-to-Image & Text-to-Video

- **Text-to-Image:** Prompt → Text Encoder → DiT Denoising → VAE Decoder → Image.
- **Text-to-Video:** Image 생성에 temporal dimension 추가.
- **한계:** Resolution과 frame 수가 증가할수록 sequence length와 연산량 증가.
- **사례:** Sora, video와 audio를 함께 생성하는 Veo 3.

## 11. Diffusion Distillation & Generalized Diffusion

- **Distillation:** Sampling steps를 줄여 inference 속도 개선. Step 감소와 생성 품질 사이 trade-off.
- **Generalized Diffusion:** Noise 혼합 방식, noise schedule, prediction target을 다양하게 설정.
- **Prediction Target:** Clean data(x), noise(ε), velocity(v).
- **수학적 관점:** Latent Variable Model, Score Function, SDE로 해석 가능.

## 12. Autoregressive Models Strike Back

= Raw pixel 대신 **압축된 discrete latent token**을 순차 예측하는 방식.

- Image → Discrete latent → Autoregressive sampling → Image 복원.
- Pixel-level 생성보다 짧은 sequence를 사용해 효율성 개선.

VAE, GAN, Diffusion, Autoregressive는 서로 배타적인 모델이 아니며 하나의 생성 pipeline에서 결합 가능.