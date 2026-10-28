# Lecture 8

개념: Transformer

### 1. Impact of Transformers

![스크린샷 2026-10-28 오후 2.18.01.png](Lecture%208/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-10-28_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.18.01.png)

- `Transformer`: Vaswani et al.(2017)의 Attention Is All You Need 논문에서 제안된 Attention 기반 신경망 구조
- 기존 RNN의 순차적 계산에 의존하지 않고, `Attention`을 핵심 연산으로 활용
- Transformer의 장점 (+Attention 구조)
    - 기계 번역에서의 **높은 성능 & 효율적인 학습**
    - NLP뿐 아니라 이미지 분류, 단백질 구조 예측 등 다양한 분야에서 활용 가능
- `Scaling Laws`: Model Size, Training Data, Compute의 확장과 모델 성능 사이에 관찰되는 관계를 설명
    - 이 3가지를 적절히 확장할 때 언어 모델의 성능이 일정한 경향에 따라 개선된다

+) `Transformer`는 Attention이 핵심이지만, 실제 구조에는 Feed Forward Network, Positional Encoding 등 다른 요소도 필요

### 2. From RNNs to Attention-Based Models

- `RNN`은 이전 Hidden State에 의존하여 입력 시퀀스를 순차적으로 처리
    
    ![스크린샷 2026-10-28 오후 2.17.34.png](Lecture%208/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-10-28_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.17.34.png)
    
    - h_t = f(h_(t-1), x_t)
- 기존 RNN 기반 Seq2Seq 모델은 Attention을 결합하여 Encoder의 정보를 참조할 수 있었지만, **순차적 계산이라는 한계가 존재**
- `Transformer`의 주요 설계 목표 세가지
    1. Computational Complexity
        - Self-Attention은 `O(n^2d)`, RNN은 `O(nd^2`의 Layer당 계산 복잡도를 가진다. n <d 일 때 `Self-Attention`이 유리
    2. Path Length
        
        ![스크린샷 2026-10-28 오후 2.17.10.png](Lecture%208/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-10-28_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.17.10.png)
        
        - RNN의 최대 경로 길이 = `O(n)`
        - Self-Attention의 최대 경로 길이 = `O(1)`
            
            → Self-Attention은 멀리 떨어진 단어도 직접 상호작용 가능
            
    3. Parallelizability
        - RNN은 이전 Hidden State가 필요해 시점 간 병렬화가 어려움
            - h_t = f(h_(t-1), x_t)
        - Self-Attention은 시퀀스 내 여러 토큰의 연산을 병렬 수행 가능
            - 이전 Hidden State를 사용하는 대신 멀리 떨어진 단어와 직접 상호작용

`Self-Attention`: 각 단어가 같은 시퀀스의 다른 단어들과의 관련성을 계산하여 정보를 결합하는 메커니즘

![스크린샷 2026-10-28 오후 2.16.56.png](Lecture%208/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-10-28_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.16.56.png)

### 3. Understanding the Transformer Model

#### 3-1) Self-Attention

- `Self-Attention`: 동일한 시퀀스 내 단어 간 관련성을 계산하고, 관련 정보를 반영하여 새로운 표현을 생성하는 메커니즘
- 각 단어의 입력 벡터로부터 학습 가능한 가중치 행렬을 사용해 `Query`, `Key`, `Value`를 생성
    - `Query (Q)`: 현재 단어가 참조하려는 정보를 나타내는 벡터
    - `Key (K)`: Query와 비교하여 관련성을 계산하는 벡터
    - `Value (V)`: Attention Weight에 따라 실제 결합되는 정보 벡터

### 계산 과정

![스크린샷 2026-10-28 오후 2.33.56.png](Lecture%208/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-10-28_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.33.56.png)

1. Q, K, V 생성: Q = XW^Q, K = XW^K, V = XW^V
2. Attention Score: E = QK^T
3. Attention Weight: A = softmax(E)
4. Output: Z=AV

![스크린샷 2026-10-28 오후 2.34.04.png](Lecture%208/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-10-28_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.34.04.png)

Attention(Q, K, V) = softmax(QK^T)V

주의) Self-Attention의 Output은 Attention Score 자체가 아니라, **Attention Weight를 적용해 Value들을 가중합한 새로운 벡터 !!!!**

#### 3-2) Feed Forward Network (FFN)

- Self-Attention으로 결합된 표현 + 비선형 변환 적용 ⇒ 모델 표현력 상승
- 각 토큰의 표현에 동일한 FFN을 독립적으로 적용

`FFN(x)=W_2 * ReLU(W_1x+b_1)+b_2`

#### 3-3) Residual Connection

- Sublayer의 Output에 기존 입력을 더하여 원래 정보를 보존하고 Gradient 전달을 도움

`y=x+F(x)` ⇒ 여기에서 x가 바로 기존 입력

#### 3-4) Layer Normalization (LayerNorm)

- 토큰의 Feature Dimension을 기준으로 정규화하여 학습을 안정화
- 원본 Transformer에서는 `Add & Norm` 구조를 사용
    - `Add & Norm` 구조 = Residual Connection + LayerNorm

`LayerNorm(x+Sublayer(x))`

#### 3-5) Scaled Dot-Product Attention

- Q와 K의 차원이 커질수록 내적값이 커질 수 있으므로 **루트(d_k)로 나누어** Scaling 수행 (그냥 기존 Attention에 루트(d_k)로 나눈다고 생각
    - d = Key의 차원
- Softmax의 과도한 포화를 완화하여 학습을 안정화

![스크린샷 2026-10-28 오후 2.55.30.png](Lecture%208/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-10-28_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.55.30.png)

#### 3-6) Positional Encoding

- Positional Encoding은 Self-Attention에 단어의 순서 정보가 내재되어 있지 않다는 문제를 해결하기 위해 사용
- 각 단어의 위치를 벡터로 표현하고 Word Embedding에 더하여 위치 정보를 반영
- 위치 정보가 추가된 입력을 사용하여 Self-Attention의 Q, K, V를 계산한다.

`z_i=x_i+p_i` ⇒ 여기서 `p_i` 가 바로 위치 정보

#### 3-7) Sinusoidal Positional Encoding

- 원본 Transformer는 서로 다른 주기의 `sin`, `cos` 함수를 활용해 위치 벡터를 생성
- 각 위치는 Embedding과 동일한 차원의 벡터로 표현됨
- 별도의 학습 가능한 위치 벡터가 필요하지 않지만, 긴 시퀀스로의 일반화에는 한계가 존재

#### 3-8) Absolute vs. Relative Positional Encoding

- **Absolute** Positional Encoding: 각 단어의 **절대적인 위치 정보**를 표현
- **Relative** Positional Encoding: 단어 사이의 **상대적인 거리와 방향 정보**를 표현한다.
- Relative Positional Encoding은 단어 간 위치 관계를 Attention 계산에 반영

#### 3-9) Multi-Head Attention

- Multi-Head Attention: 여러 Attention Head를 병렬로 계산하고 결과를 결합하는 방식
- 각 Head는 서로 다른 학습 가능한 Q, K, V 가중치 행렬을 사용
    
    ⇒ 여러 표현 공간에서 다양한 단어 간 관계를 학습 가능
    

**계산 과정**

1. 각 Head마다 독립적으로 Q, K, V 생성
2. 각 Head에서 Scaled Dot-Product Attention 계산
3. 모든 Head의 Output을 Concatenation
4. Linear Projection을 적용하여 최종 Output 생성

`head_i=Attention(Q_i,K_i,V_i)`

`MultiHead(X)=Concat(head_1,...,head_h)W^O`

#### 3-10) Transformer Encoder

- `Input Embedding + Positional Encoding → Multi-Head Self-Attention → Add & Norm → Feed Forward → Add & Norm`
- Encoder Layer를 여러 번 반복하여 입력 문장의 문맥적 표현을 학습

(원본 Transformer는 Encoder Layer를 6개 사용)

#### 3-11) Decoder

- Decoder: 이전까지의 출력 토큰과 Encoder의 정보를 활용하여 다음 토큰을 예측
- `Masked Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm`으로 구성된 Layer 반복

#### 3-12) Masked Self-Attention

- 미래 토큰을 미리 참조하지 못하도록 Attention Mask를 적용
    - 방법: 참조할 수 없는 위치의 Attention Score를 -무한대로 설정 → Softmax 이후 Weight가 0이 되도록 함
- 학습 시 미래 정보 누출을 방지하면서 병렬 계산을 가능하게 함

#### 3-13) Cross-Attention

- Query: Decoder에서 생성
- **Key, Value: Encoder에서 생성**
    
    ⇒ Decoder가 입력 문장의 문맥 정보를 참조하도록 함
    

#### 3-14) Output Generation

- Decoder의 최종 출력을 Linear Layer로 Vocabulary Size 차원의 Logits로 변환
- Softmax로 다음 토큰에 대한 확률 분포를 생성
- 추론 시에는 이전에 생성한 토큰을 바탕으로 다음 토큰을 순차적으로 생성

### 4. **Drawbacks and Variants of Transformers**

#### 4-1) Transformer의 한계

- **Quadratic Complexity**
    
    Self-Attention은 모든 단어 쌍의 관계를 계산 → Sequence Length에 따라 계산량이 `O(n^2d)`로 증가
    
- **Positional Encoding**
    
    단순한 절대 위치 정보만으로는 단어 간 관계를 충분히 표현하기 어려울 수 있음
    

#### 4-2) 개선 방법

- `Linformer`: Key, Value의 **시퀀스 길이 차원을 축소**하여 Attention 계산 비용 줄임
- `BigBird`: Local, Global, Random Attention을 조합하여 일부 토큰 간 관계만 계산
- `Relative Positional Encoding`: 단어 간 상대적 위치 관계를 반영(절대적X)
- `Rotary Positional Embeddings (RoPE)`: 회전 변환을 활용한 위치 정보를 Attention에 반영

**정리**

- Transformer는 병렬 처리에 유리하지만 긴 시퀀스에 대한 계산 비용이 크다는 한계가 존재
- 다양한 개선 방법이 제안되었으나, 모든 변형이 기존 Transformer보다 높은 성능을 보장하지는 않음