# 위키독스 07

개념: 임베딩과 벡터 스토어

## 1. 임베딩(Embedding)

임베딩은 **텍스트의 의미적 특징을 숫자 배열인 벡터로 표현하는 기술**

- 글자가 달라도 의미적 관련성 존재
    - ‘강아지’와 ‘반려견’은 표현이 다르지만 의미적으로 관련 O
    - 임베딩 모델은 학습을 통해 이런 텍스트를 유사한 벡터로 표현

```
텍스트 → 임베딩 모델 → 숫자 벡터
```

컴퓨터는 생성된 벡터를 수학적으로 비교하여 텍스트 사이의 의미적 관련성을 계산

- 벡터의 각 숫자에 ‘동물’, ‘귀여움’ 같은 의미를 사람이 직접 지정하는 것 X
- 질문과 문서를 비교하려면 **같은 벡터 공간에서 비교할 수 있도록 호환되는 임베딩 모델을 사용해야 한다!!**
- 벡터의 차원이 같다는 것만으로 서로 비교 가능한 것은 아님
    - 차원이 같으면 숫자 계산은 가능하지만, 그 결과를 의미의 유사도로 해석하려면 숫자들이 같은 좌표 체계를 사용해야 함

## 2. 벡터의 차원(Dimension)

차원: **벡터 하나에 포함된 숫자의 개수**

```
vector = [0.2, -0.1, 0.8]  # 숫자 3개 → 3차원 벡터
```

+) 문서 10개를 각각 768차원 벡터로 변환한 경우

| 코드 | 의미 | 값 |
| --- | --- | --- |
| `len(doc_vectors)` | 문서 벡터의 개수 | `10` |
| `len(doc_vectors[0])` | 첫 번째 벡터의 차원 | `768` |

문서 개수(`len(doc_vectors)`) ≠  벡터의 차원(`len(doc_vectors[0])`)

## 3. 코사인 유사도(Cosine Similarity)

코사인 유사도: **두 벡터가 얼마나 같은 방향을 가리키는지** 측정하는 지표

![스크린샷 2026-10-31 오후 11.13.30.png](%EC%9C%84%ED%82%A4%EB%8F%85%EC%8A%A4%2007/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-10-31_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_11.13.30.png)

- 분자 A · B: 같은 위치의 숫자를 곱한 뒤 모두 더한 값인 **내적**
- ||A||, ||B||: 각 벡터의 길이

벡터 길이의 영향을 제거하여 방향을 비교 (값의 범위: -1 ~ 1)

- `1` = 같은 방향
- `0` = 서로 직각
- `-1` = 정반대 방향

임베딩 검색에서는 일반적으로 코사인 유사도가 높을수록 의미적으로 관련 있다고 판단

주의할 점

- `0.9`는 ‘의미가 90% 같다’거나 ‘정답일 확률이 90%’라는 뜻 X
- 음수라고 반드시 반대말이라는 뜻 X
- 질문과의 유사도가 높아도 문서 내용이 사실이라는 보장 X

## 4. LangChain의 임베딩 메서드

`LangChain`은 문서와 질문을 임베딩하는 메서드를 구분

| 메서드 | 입력 | 출력 |
| --- | --- | --- |
| `embed_documents(texts)` | 문서 문자열 리스트 | 문서별 벡터를 담은 리스트 |
| `embed_query(text)` | 질문 문자열 하나 | 질문 벡터 하나 |

```
texts = [
    "반려견에게 초콜릿을 급여하면 위험합니다.",
    "자동차 엔진을 정기적으로 점검해야 합니다.",
]

# 문서 2개를 각각 벡터로 변환
doc_vectors = embeddings.embed_documents(texts)

# 질문 하나를 벡터로 변환
query_vector = embeddings.embed_query(
    "강아지가 먹으면 안 되는 음식은?"
)

# 반환값의 구조
doc_vectors = [
    [첫 번째 문서의 벡터값들],
    [두 번째 문서의 벡터값들],
]

query_vector = [질문의 벡터값들]
```

- `embeddings`: 텍스트를 숫자 벡터로 변환하는 임베딩 모델 객체
- `embed_documents()`: 문서를 벡터로 변환하는 역할 (반환된 벡터의 순서는 입력 문서의 순서에 대응)
- `embed_query()`: 질문 문자열 하나를 받아 질문 자체의 벡터를 반환
- 생성된 벡터들은 서로 비교 가능한 공간에 존재
- 두 메서드 모두 벡터를 생성하는 역할, 호출만으로 문서 저장/유사도 검색/ 답변 생성까지 수행 X

## 5. 질문 임베딩·문서 검색·답변 생성의 차이

| 단계 | 수행하는 작업 | 결과 |
| --- | --- | --- |
| `질문 임베딩` | 질문을 벡터로 변환 | 질문 벡터 |
| `문서 검색` | 질문 벡터와 저장된 문서 벡터의 유사도 비교 | 관련 문서 |
| `답변 생성` | 질문과 검색된 문서의 원문을 LLM에 전달 | 자연어 답변 |

```
[문서 준비]
문서 → 문서 벡터 생성 → 원문과 연결하여 저장

[질문 처리]
질문 → 질문 벡터 생성
           ↓
    문서 벡터들과 유사도 비교
           ↓
    관련 문서의 원문 가져오기
           ↓
    질문 + 문서 원문 → LLM → 답변
```

- 일반적인 텍스트 RAG 기준 **1) 벡터를 이용해 문서를 찾고, 2) 찾은 문서의 원문 텍스트를 LLM에 전달**
- 검색용 벡터를 그대로 LLM에 전달하거나 벡터에서 원문을 복원하는 방식 X

## **6. 벡터 스토어(Vector Store)**

임베딩 벡터를 저장하고, 질문 벡터와의 유사도를 이용해 관련 문서를 검색하는 역할

기능

- 검색된 문서의 원문을 가져올 수 있도록 벡터와 원문 사이의 연결을 유지
- 문서에 출처, 페이지 번호, 작성일 등의 메타데이터를 함께 연결 가능
- 질문과 유사한 문서를 검색

추가로 일반적인 RAG에서는 검색용 벡터에서 원문을 복원하지 않고, 해당 벡터에 연결된 원문을 가져와 LLM에 전달한다

| 구성 요소 | 역할 |
| --- | --- |
| `임베딩 모델` | 문서와 질문을 벡터로 변환 |
| `벡터 스토어` | 벡터 저장 및 관련 문서 검색 |
| `LLM` | 검색된 원문을 참고해 답변 생성 |

## **7. 모델 사용**

### 7-1) OpenAI 모델 사용

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

query_vector = embeddings.embed_query("강아지에게 위험한 음식은?")
```

- `OpenAIEmbeddings(...)`: 사용할 모델과 API 연결 설정을 담은 객체를 준비
- `embed_query(...)`: 질문 텍스트를 API로 전송하여 실제 임베딩을 요청
- 임베딩 모델은 OpenAI 서버에서 실행되며, 반환받은 벡터는 내 프로그램에서 사용
    - 흐름
        
        ```json
        내 프로그램
            ↓ 질문 텍스트를 API로 전송
        OpenAI 서버의 임베딩 모델
            ↓ 질문을 벡터로 변환
        내 프로그램이 벡터를 받아 query_vector에 저장
        ```
        
- 디폴트 설정: `text-embedding-3-small`은 1,536차원 벡터를 반환

### **7-2) Hugging Face 모델 사용 (로컬 실행 ver)**

패키지 설치: `pip install langchain-huggingface sentence-transformers`

```python
from langchain_huggingface import HuggingFaceEmbeddings

# 모델 다운로드 및 메모리에 로드
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 내 컴퓨터에서 질문을 벡터로 변환
query_vector = embeddings.embed_query(
    "강아지가 먹으면 안 되는 음식은?"
)

print(len(query_vector))  # 384
```

- `HuggingFaceEmbeddings`를 사용하면 모델을 다운로드하여 코드를 실행하는 컴퓨터에서 임베딩을 계산 가능
- 객체를 준비할 때 필요한 모델 파일을 다운로드하고 메모리에 로드.
- `embed_documents()`와 `embed_query()`를 호출하면 로컬 모델이 텍스트를 벡터로 변환
- 임베딩 모델을 교체하면 출력 차원이 같더라도 좌표 체계가 다를 수 있다. 따라서 일반적으로 기존 문서도 새 모델로 다시 임베딩해야 함
- 추가로 `Hugging Face` 모델을 서버 API로 사용하는 방법도 존재

### 7-3) FAISS로 문서 저장 및 검색하기

```python
from langchain_community.vectorstores import FAISS

# 문서 임베딩 및 스토어 생성
vector_store = FAISS.from_texts(
    texts,
    embedding=embeddings,
)

# 관련 문서 검색
results = vector_store.similarity_search(
    "강아지가 먹으면 안 되는 음식은?",
    k=1,
)

# 첫 번째 검색 결과의 원문
print(results[0].page_content)
```

- `from_texts()`: 내부적으로 `embed_documents()`를 호출하여 문서 벡터를 생성하고, 원문과 연결하여 저장
- `similarity_search()`: 내부적으로 `embed_query()`를 호출하여 질문을 벡터로 변환한 뒤 관련 문서 검색 수행
- `k`: 반환할 문서 수
- 반환값은 `Document` 객체들의 리스트이며, `.page_content`로 원문 확인 가능
- 검색 결과는 저장된 문서이며, LLM이 새롭게 생성한 답변이 아니다!
- 디폴트 설정: `L2` 거리

**FAISS에서 문서 벡터 값 저장 및 불러오기**

```python
# 현재 스토어를 faiss_index 폴더에 저장
vector_store.save_local("faiss_index")

# 저장할 때와 호환되는 임베딩 모델 객체를 먼저 준비
loaded_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

results = loaded_store.similarity_search(
    "강아지가 먹으면 안 되는 음식은?",
    k=1,
)
```

```python
저장한 파일에서 불러오기 → 기존 문서 벡터 재사용
새 질문 입력           → 임베딩 모델로 질문 벡터 생성
                                  ↓
                         기존 문서 벡터와 비교
```

- `save_local("faiss_index")`: 검색 인덱스, 문서 원문, 연결 정보를 파일로 저장
- `load_local(...)`로 불러오기: 기존 문서를 다시 임베딩하지 않고 저장된 벡터를 재사용
- 불러올 때 전달하는 `embeddings`는 새로운 질문을 벡터로 변환할 때 사용됨
    - 기존 문서 벡터와 호환되는 모델이어야 함
    - 임베딩 모델 자체가 스토어 파일에 함께 저장되는 것은 아니기 때문에 `embeddings`를 함께 전달하는 것
- `allow_dangerous_deserialization=True`
    - 저장된 문서 정보를 읽을 때 사용하는 pickle 형식의 복원을 허용하는 설정
    - 직접 만들거나 변조되지 않은 파일 등 신뢰하는 파일에만 사용해야 함!!

### **7-4) Chroma 기본 사용법과 데이터 저장**

패키지 설치 명령어: `pip install langchain-chroma`

```python
from langchain_chroma import Chroma

# embeddings는 미리 준비한 임베딩 모델 객체
vector_store = Chroma(
    collection_name="study_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
```

| 설정 | 역할 |
| --- | --- |
| `collection_name` | 문서를 묶어 관리할 컬렉션 이름 |
| `embedding_function` | 문서와 질문을 벡터로 변환할 모델 |
| `persist_directory` | 데이터를 파일로 보관할 폴더 |
- `add_documents(documents)`: 문서 임베딩 후 추가
- `similarity_search(question, k=...)`: 관련 문서 검색 수행
- 검색 결과는 `Document` 객체들의 리스트, `.page_content`로 원문, `.metadata`로 부가 정보 확인 가능
- 저장 폴더를 지정 시 (`persist_directory`) 프로그램을 종료한 뒤에도 데이터 재사용 가능
- 다시 실행할 때 같은 저장 폴더와 컬렉션으로 연결하면 됨
    - 기존 문서를 검색하기 위해 다시 추가할 필요 X
- 다시 연결할 때도 기존 문서 벡터와 호환되는 임베딩 모델을 사용해야 함!!

## 8. **유사도 검색과 MMR**

- 유사도 검색: 질문과의 유사도가 높은 문서를 우선 선택하는 것
    
    → 그러나 비슷한 내용을 담은 문서가 많으면 검색 결과에 중복 정보가 포함될 수 있음
    
- **MMR(Maximal Marginal Relevance)**
    - 질문과의 관련성과, 이미 선택한 문서들과의 중복 정도를 함께 고려하는 방법
    - 목적: 관련 있는 정보를 가져오면서 **검색 결과의 중복을 줄이는 것**
    - 단, MMR 자체가 최소 관련성을 보장하거나 무관한 문서를 반드시 제외하는 것은 아님

### 8-1) **MMR 검색의 주요 설정**

```python
results = vector_store.max_marginal_relevance_search(
    question,
    k=4,
    fetch_k=30,
    lambda_mult=0.2,
)
```

- `fetch_k` : 먼저 검색하여 MMR에 전달할 후보 문서 수
- `k` : 최종적으로 선택할 문서 수
- `lambda_mult` : 관련성과 다양성의 균형을 조절하는 값
    - 범위: `0 ~ 1`
    - 1에 가까울수록 질문과의 관련성, 0에 가까울수록결과의 다양성을 강조
    - `0.5`는 관련성과 중복 억제 항의 가중치를 동일하게 둔다는 의미
- **후보 문서가 부족하면** 최종 결과는 `k`개보다 적을 수 있음
- 단, MMR 자체가 최소 관련성을 보장하는 것은 아니다.

### 8-2) **임계값 검색(Threshold Search)**

- 관련성 점수가 설정한 임계값 이상인 문서만 반환하는 검색 방식

```python
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.8,
    },
)
```

- `k`: 반환할 최대 문서 수
- `score_threshold`: 최소 관련성 기준을 지정(임계값)
    - 이 값보다 낮은 점수를 가진 문서는 최종 후보여도 반환되지 않음
- 조건을 만족하는 문서가 부족하면 `k`보다 적게 반환하며, 모두 기준 미달이면 빈 결과가 반환될 수 있음
- 단, 임계값을 통과했다고 문서의 내용이 사실이라는 뜻은 아님
- 적절한 임계값은 임베딩 모델과 점수 계산 방식, 실제 검색 결과를 바탕으로 정해야 함

### 8-3) **Retriever**

- 질문 문자열을 입력받아 관련 `Document` 객체들의 리스트를 반환하는 공통 인터페이스

```python
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.8,
    },
)
```

- 벡터 스토어의 `as_retriever(...)`로 검색 전략과 세부 조건을 설정한 Retriever를 준비 가능 (기존 벡터 스코어 활용 가능)
- `retriever.invoke(question)`가 실제 검색 실행
- Retriever를 만드는 것만으로 검색이 실행되거나 문서가 별도 스토어에 복사되지는 않는다
- 조건을 통과한 문서가 없으면 빈 리스트 `[]`를 반환할 수 있다
    - 사용자 안내 문구는 별도 코드로 처리가 필요
    
    ```python
    # 예시
    documents = retriever.invoke(
        "강아지가 먹으면 안 되는 음식은?"
    )
    
    if not documents:
        print("기준을 충족하는 문서가 없습니다.")
    else:
        for doc in documents:
            print(doc.page_content)
    ```
    
- Retriever 자체는 LLM 답변 생성 X

**호출 별 invoke 정리**

- `model.invoke(...)` : LLM 응답 생성
- `tool.invoke(...)` : 도구 실행
- `retriever.invoke(...)` : 관련 문서 검색

## 9. **LCEL로 검색과 원문 정리 연결하기**

`LCEL(LangChain Expression Language)`: 여러 처리 단계를 연결하는 방식

- `|`: 앞 단계의 출력을 다음 단계의 입력으로 전달한다는 의미의 기호

```python
def format_docs(documents):
    return "\n\n".join(
        doc.page_content for doc in documents
    )

retrieval_chain = retriever | format_docs

context = retrieval_chain.invoke(question)
```

```
# 실행 흐름
질문 문자열
    ↓ retriever
검색된 Document 리스트
    ↓ format_docs
원문들을 합친 문자열
```

- `format_docs()`: 각 문서의 `.page_content`를 꺼내 빈 줄로 구분하여 합침
    - 체인을 정의하는 것만으로 실행되지는 않음
- `invoke()`: 호출 시 연결된 단계들이 실행됨
- 체인의 최종 반환값 = 마지막 단계의 출력
- 예시 코드의 체인에는 LLM X → LLM 답변 생성 X

```
# 최종 전체 흐름
질문 → 문서 검색 → 원문 합치기
                       ↓
             질문과 원문을 프롬프트에 넣기
                       ↓
                  LLM 답변 생성
                       ↓
                 답변 텍스트 추출
```

- `retriever`: 검색 조건에 따라 가져올 문서들을 선택
- `format_docs`: 선택된 문서들의 원문을 꺼내 하나의 문자열로 합치고, 결과를 `context`에 담음
- 프롬프트: `context`와 사용자 질문을 함께 넣어 LLM에 전달

### 9-1) **프롬프트, LLM, 출력 파서 연결**

```python
answer_chain = prompt | model | StrOutputParser()

answer = answer_chain.invoke({
    "context": context,
    "question": question,
})
```

**구성 요소**

- `prompt` : 참고 문서와 질문을 넣어 메시지 구성
- `model` : 메시지를 받아 답변 생성
- `StrOutputParser()` : 모델 응답에서 텍스트를 추출하여 문자열로 반환

**프롬프트 입력 변수**

- `context` : 검색된 문서 원문들을 합친 문자열
- `question` : 사용자의 질문
- 딕셔너리의 키가 프롬프트의 `{context}`, `{question}`에 대응

**코드**

- `answer_chain.invoke()`를 실행하면 체인 안의 모델도 실행됨
    - 따라서 `model.invoke()`를 별도로 호출할 필요 X
- 마지막 단계가 `StrOutputParser()` → 최종 반환값 = 문자열

검색부터 답변 생성까지 하나의 체인으로 연결하는 경우의 흐름

```python
                      ┌→ retriever → format_docs → context ┐
사용자 질문 ──────────┤                                    ├→ prompt → model → 문자열
                      └→ RunnablePassthrough → question ───┘
```

### 9-2) **LCEL로 검색부터 답변 생성까지 연결하기**

가정) `retriever`, `prompt`, 채팅 모델 `model`가 준비됨

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(documents):
    return "\n\n".join(
        doc.page_content for doc in documents
    )

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke(
    "강아지가 먹으면 안 되는 음식은?"
)
```

**입력을 준비하는 두 경로**

- `context` : 질문 → 문서 검색 → 원문들을 하나의 문자열로 합치기
- `question` : 사용자 질문을 변경하지 않고 그대로 전달 (`RunnablePassthrough()` 이용)

```
# 두 경로의 결과가 프롬프트에 전달되는 형태
{
    "context": "검색된 문서 원문들을 합친 문자열",
    "question": "사용자의 질문",
}
```

**이후 실행 과정**

1. `prompt`: `{context}`와 `{question}`에 값을 채워 메시지를 구성
2. `model`: 참고 문서와 질문을 받아 답변을 생성
3. `StrOutputParser()`: 모델 응답에서 텍스트를 추출하여 문자열로 반환

**참고 사항**

- 체인을 정의하는 것만으로는 실행되지 않는다.
- `rag_chain.invoke(question)`를 호출하면 연결된 단계들이 실행됨
- `model.invoke()`를 별도로 작성하지 않아도 체인 내부에서 모델이 실행됨
    - `model` 단계
- 위 체인은 검색 결과가 없어도 모델을 호출하기 때문에 문서가 없을 때 호출을 중단하려면 별도 조건 분기를 작성하면 됨

## 10. 복습용 정리

**1. 구성 요소의 역할**

- 임베딩 모델: 문서와 질문을 벡터로 변환
- 벡터 스토어: 벡터를 저장하고 원문과 연결하여 검색
- Retriever: 질문을 받아 검색 조건에 맞는 문서 리스트 반환
- `format_docs` : 검색된 문서들의 원문을 문자열로 합침
- 프롬프트: 참고 원문과 질문을 넣어 메시지 구성
- 채팅 LLM: 메시지를 받아 답변 생성
- `StrOutputParser` : 모델 응답에서 텍스트를 추출해 문자열로 반환

**2. 벡터 비교의 핵심**

- 차원은 벡터 하나에 포함된 숫자의 개수이다.
- 차원이 같아도 좌표 체계가 다르면 의미 있는 비교가 되지 않는다.
- 질문과 문서는 같은 벡터 공간을 사용하는 호환되는 모델로 임베딩해야 한다.
- 코사인 유사도는 벡터의 방향을 비교한다.
- 유사도 점수는 정답 확률이나 문서의 사실 여부를 나타내지 않는다.

**3. 저장과 재사용**

- `embed_documents()`는 문서별 벡터들을, `embed_query()`는 질문 자체의 벡터 하나를 반환한다.
- FAISS의 `from_texts()`는 문서 임베딩과 스토어 생성을 함께 처리한다.
- FAISS는 `save_local()`과 `load_local()`로 저장·불러오기를 수행할 수 있다.
- Chroma는 `persist_directory`로 저장 폴더를, `collection_name`으로 문서 묶음을 지정한다.
- 저장된 문서 벡터는 재사용하지만, 새로운 질문은 벡터로 변환해야 한다.

**4. 검색 전략**

1. 유사도 검색: 질문과 가까운 문서를 우선 선택
2. MMR: 관련성을 고려하면서 결과의 중복을 줄임
3. 임계값 검색: 최소 관련성 기준을 통과한 문서만 반환

검색 파라미터 정리

- `k`: 최종 반환할 최대 문서 수
- `fetch_k`: MMR에 전달할 후보 문서 수
- `lambda_mult`: 높을수록 관련성, 낮을수록 다양성 강조
- `score_threshold`: 관련성 점수의 통과 기준

**5. 전체 실행 흐름**

```
질문
 ↓
질문 임베딩 → 저장된 문서 벡터와 비교
 ↓
검색 전략에 따라 문서 선택
 ↓
연결된 원문을 가져와 문자열로 합침
 ↓
참고 원문 + 질문으로 프롬프트 구성
 ↓
LLM 답변 생성
 ↓
답변 문자열 추출
```

- 벡터에서 원문을 복원하는 것이 아니라, 벡터와 연결된 원문을 가져온다.
- LCEL의 `|`는 앞 단계의 출력을 다음 단계의 입력으로 전달한다.
- `RunnablePassthrough()`는 입력을 그대로 전달한다.
- `invoke()`는 실행 대상에 따라 검색·도구 실행·LLM 호출 등을 수행한다.
- 검색 결과가 비어도 체인이 자동으로 중단되지는 않는다. LLM 호출을 막으려면 별도 조건 분기가 필요하다.