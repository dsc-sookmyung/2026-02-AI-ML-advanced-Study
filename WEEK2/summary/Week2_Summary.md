## 6강. Sequence to Sequence Model

### 📒 강의 개요

언어 모델 평가 척도인 Perplexity(혼잡도), 바닐라 RNN의 기울기 소실(Vanishing Gradient) 및 폭발(Exploding Gradient) 한계와 이를 극복하기 위한 장단기 메모리(LSTM) 구조를 학습함. 또한 심층(Deep/Stacked) 및 양방향(Bidirectional) RNN의 활용 방식과 Sequence-to-Sequence(인코더-디코더) 기반의 신경망 기계 번역(NMT) 기본 원리를 다룸.

### 핵심 개념 및 용어

1. **Perplexity (혼잡도)**

* 언어 모델이 생성하거나 평가한 테스트 텍스트의 조건부 확률에 역수를 취한 후 기하평균을 낸 값으로, 교차 엔트로피(Cross-Entropy) 손실의 지수값($\exp(CE)$)과 같음

* 매 시점 다음 단어를 맞추기 위해 굴려야 하는 '균등한 면을 가진 주사위의 면 수'(선택지의 수)로 비유할 수 있으며, 값이 낮을수록 모델의 예측 성능이 우수함

1. **RNN → 기울기 소실과 폭발**

* 기울기 소실 (Vanishing Gradient)

  * 역전파(Backpropagation through time) 과정에서 가중치 행렬 $W$가 반복 곱해지며 시점 간 거리가 멀어질수록 역전파되는 그래디언트가 기하급수적으로 작아져 소멸함.

  * 따라서 기본 RNN은 먼 문맥 정보 (Long distance dependency)를 학습에 제대로 반영하지 못함

* 기울기 폭발 (Exploding Gradient) 및 grdient clipping

  * 가중치 행렬의 고윳값이 커서 그래디언트가 기하급수적으로 폭증하면 파라미터가 크게 튀어 학습이 발산하거나 수치적 오류(NaN/Inf)가 발생

  * ⇒ 따라서 계산된 그래디언트의 L2 Norm이 특정 임계값(Threshold)을 초과할 경우, 방향은 유지한 채 크기를 강제로 축소시키는 **Gradient Clipping** 기법을 적용

1. **LSTM (Long Short-Term Memory)**

* 매 시점 이전 정보를 행렬곱으로 덮어쓰는 바닐라 RNN과 달리, 셀 상태(Cell State, $c_t$)를 두고 정보를 **선형적 덧셈** 형태로 누적하여 기울기 소실 문제를 완화

* 3개의 Gate 구조 (Sigmoid 함수)

  * 망각 게이트 (Forget Gate, $f\_t$) : 이전 셀 상태($c\_{t-1}$)에서 얼마만큼의 정보를 유지하거나 잊을지 결정

  * 입력 게이트 (Input Gate, $i\_t$) : 현재 입력과 은닉 상태로부터 계산된 새로운 후보 정보 중 얼마를 셀에 추가할지 결정

  * 출력 게이트 (Output Gate, $o\_t$) : 셀 상태의 기억 중 현재 시점의 출력 및 은닉 상태($h\_t$)로 전달할 정보의 비율을 결정

1. **Bidirectional & Stacked RNN**

* 양방향 RNN (Bidirectional RNN)

  * 문장을 앞에서 뒤로 읽는 순방향 RNN과 뒤에서 앞으로 읽는 역방향 RNN의 은닉 상태를 연결(Concatenate) ⇒ 양쪽 전체 문맥을 모두 고려한 단어 표현을 생성

  * 텍스트 생성 디코더에는 부적합하며 문맥 인코딩에 활용

* **적**층 RNN (Stacked / Deep RNN)

  * 시간 축(가로)뿐만 아니라 수직(세로)으로 여러 층의 RNN 레이어를 쌓아 더 고수준의 계층적 특징(Feature)을 추출

1. **신경망 기계 번역 (NMT), Seq2seq**

* 인코더-디코더(Encoder-Decoder) 구조

  * encoder : 입력 소스 문장을 읽어들이며 전체 문장의 의미를 요약한 최종 은닉 상태 벡터를 생성

  * decoder : 인코더의 최종 은닉 상태를 초기 상태로 전달받아, 타깃 언어의 단어들을 조건부 언어 모델(Conditional Language Model) 방식으로 한 단어씩 순차적으로 생성

* End-to-End 학습

  * 과거 통계 기반 번역(SMT)과 달리 단일 인공신경망 전체를 Cross-Entropy Loss에 대해 End-to-End 역전파로 동시에 학습 ⇒ 번역 품질 향상

## 👀 새롭게 배운 점

가변 길이의 시퀀스 입력을 인코더 RNN을 통해 단일 고정 차원의 Context Vector로 압축한 뒤, 디코더 RNN이 이를 바탕으로 순차적인 타깃 시퀀스를 생성해 내는 Seq2Seq 구조의 핵심 매커니즘을 배웠습니다. 또한 번역이나 요약처럼 입력과 출력의 길이가 서로 다른 복잡한 태스크를 인코더-디코더의 조건부 언어 모델링(Conditional Language Modeling)과 Teacher Forcing을 통해 정교하게 학습시키는 원리를 이해할 수 있었습니다.

## 💡 **느낀 점 (더 알아보고 싶은 내용, 어려웠던 부분 등)**

임의의 길이를 가진 문장 전체의 풍부한 정보를 하나의 고정 크기 벡터에 모두 우겨넣는 과정에서 발생하는 병목 현상과, 문장이 길어질수록 초반 정보가 희미해지는 장기 의존성(Long-term Dependency) 문제의 한계가 매우 뚜렷하게 다가왔습니다. 이러한 정보 손실을 극복하기 위해 디코더가 인코더의 모든 시점 상태를 유동적으로 직접 조회할 수 있도록 돕는 어텐션이 왜 필수적으로 등장하게 되었는지 깊이 있게 더 탐구해보고 싶습니다.