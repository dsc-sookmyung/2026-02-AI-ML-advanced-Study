# Week 1 - Summary
주제

- Lecture 1,2,3 - Intro and Word Vectors / Word Vectors and Language Models / Backpropagation and Neural Network Basics

과제

- https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/assignments/a1_preview/exploring_word_vectors.html

### 내용

## 1강. Intro and Word Vectors

### 📒 강의 개요

컴퓨터가 언어의 '의미를 어떻게 표현하는지 살펴봄, 전통적인 원-핫 벡터(One-hot vector)의 한계를 극복하기 위해 등장한 Distributional Semantics과 **Word2Vec** 알고리즘의 원리를 수학적으로 유도함.

### 핵심 개념 및 용어

- **Denotational vs Distributional Semantics**:
    - **지시적 의미론(Denotational)**: 단어(기호)를 실제 세계의 사물이나 사전적 정의에 1:1 매핑 (WordNet). 단어 간의 미묘한 뉘앙스나 관계 표현 불가.
    - **분포 의미론(Distributional)**: 특정 단어 주변에 등장하는 이웃 단어들의 문맥 정보로 단어의 의미를 규정.
- **원-핫 벡터(One-hot Vector)의 한계**: 단어 벡터 간의 내적(Dot product)이 0으로 직교(Orthogonal)하므로 단어 간 유사도(호텔과 모텔 등)를 측정할 수 없음.
- **Word2Vec (Skip-gram)**: 중심 단어(w_t)가 주어졌을 때 주변 문맥 단어(w_{t+j})가 나타날 확률을 최대화하는 방식으로 밀집 벡터(Dense Vector)를 학습.
- **Softmax 함수**: 내적 점수를 확률 분포(0 ~ 1, 총합 1)로 변환하는 활성화 함수.

## 2강. Word Vectors and Language Models

### 📒 강의 개요

Word2Vec의 최적화 기법(경사하강법 및 Negative Sampling), 단어 벡터의 선형 의미론적 성질(단어 유추), 통계 기반 GloVe 모델, 그리고 단어 벡터를 입력으로 사용하는 신경망 분류기(Neural Classifier)의 기초를 다룸.

### 핵심 개념 및 용어

- **경사하강법(GD) vs 확률적 경사하강법(SGD)**: 전체 코퍼스 계산 대신 미니배치 단위로 노이즈가 섞인 그래디언트를 계산해 학습 속도를 획기적으로 개선.
- **네거티브 샘플링(Negative Sampling)**: 전체 어휘 사전(수십만 개)에 대한 Softmax 정규화 비용을 줄이기 위해, 실제 문맥 단어(Positive) 1개와 무작위로 추출한 비문맥 단어(Negative) k개에 대해 로지스틱 회귀로 이진 분류 학습
- **벡터 연산(Analogy)**을 통한 벡터 덧셈/뺄셈을 통한 의미론적 관계 표현.
- **카운트 기반 모델과 GloVe**: Co-occurrence Matrix에 SVD를 적용하던 고전 방식과 Word2Vec의 장점을 결합하여, 공기 확률의 비율(Ratio)을 로그 선형 모델로 학습.
- **단어 다의성(Word Senses)**: 하나의 단어 벡터가 여러 의미(Sense) 벡터들의 가중 선형 결합(Superposition) 형태로 압축되어 저장됨.

## 3강. Backpropagation, Neural Network

### 📒 강의 개요

신경망 학습의 본질인 행렬 미분(Matrix Calculus)과 계산 그래프(Computation Graph) 상에서의 역전파(Backpropagation) 알고리즘을 수학 및 구현 관점에서 이해함.

### 핵심 개념 및 용어

- **비선형 활성화 함수(Non-linearities)**: 선형 변환(W x + b)만 중첩하면 전체가 하나의 선형 함수로 축약되므로, 복잡한 비선형 함수를 근사하기 위해 ReLU, Leaky ReLU, GeLU, Sigmoid, TanH 등이 필수적.
- **야코비안(Jacobian Matrix)**: 다변수 입력 벡터(n차원)에서 다변수 출력 벡터(m차원)로 가는 변환의 모든 편미분을 모아둔 m x n 행렬.
- **Shape Convention**: 수학적 야코비안은 긴 행 벡터나 거대 행렬이 될 수 있으나, 딥러닝 프레임워크 구현에서는 그래디언트가 파라미터 텐서와 동일한 차원/형태(Shape)를 유지하도록 전치(Transpose) 및 형태를 맞춤.
- **역전파 알고리즘**:
    
    체인 룰(Chain Rule)을 적용하며, 계산 그래프의 노드별 중간 결과를 캐싱하여 중복 계산을 방지.
    
- **그래디언트 검사(Numerical Gradient Checking)**: 중심 차분 공식을 통해 해석적(Analytic) 미분 코드 구현이 올바른지 수치적으로 검증.

## 👀 새롭게 배운 점

원-핫 벡터의 한계를 넘어 문맥 기반으로 단어의 의미와 관계를 고차원 공간에 표현하는 임베딩(Word2Vec)과, 비선형 활성화 함수 및 체인 룰 기반의 역전파로 동작하는 신경망의 기본 원리를 배웠습니다. 또한 복잡한 언어 이해가 주변 단어 확률 최적화와 미분이라는 단순한 수학적 원리 위에서 정교하게 구현된다는 것을 배울 수 있었습니다.

## 💡 **느낀 점 (더 알아보고 싶은 내용, 어려웠던 부분 등)**

컴퓨터가 인간의 언어를 이해하고 복잡한 추론을 수행하는 과정이 당연한 것이 아니라,  철저한 수학적 기초 위에 구축되어 있다는 점을 깨달았습니다. 단순한 확률 모델이 성별이나 국가 같은 복잡한 의미 관계를 기하학적 공간에 스스로 정렬해 내는 모습이 인상적이었습니다.