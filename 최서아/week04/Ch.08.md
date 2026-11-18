# Ch.08 RAG(검색 증강 생성) 파이프라인 정리

> 참고: https://wikidocs.net/333792  

## 1. RAG란?

**RAG(Retrieval-Augmented Generation, 검색 증강 생성)** 는 LLM이 답변하기 전에 외부 지식(문서, 사내 자료 등)을 **검색(Retrieval)** 하고, 검색된 내용을 **근거(Context)** 로 제공해 답변을 **생성(Generation)** 하는 방식이다.

- **필요성:** LLM은 최신 정보나 사내 문서를 모를 수 있으며, 사실과 다른 내용을 그럴듯하게 생성하는 **환각(Hallucination)** 문제가 있다.
- **장점:** 새로운 문서를 반영하기 쉽고, 답변의 근거를 제시할 수 있으며, 환각을 줄이는 데 도움이 된다.
- **한계:** 잘못된 문서를 검색하면 답변도 틀릴 수 있다. RAG가 환각을 완전히 없애지는 못한다.
- **Grounding:** 모델의 답변이 실제 문서나 데이터에 근거하도록 하는 것.

### RAG의 3단계

| 단계 | 작업 | 핵심 기술 |
|---|---|---|
| ① 인덱싱(Indexing) | 문서 로드 → 청크 분할 → 임베딩 → 저장 | Loader, Splitter, Embedding, Vector Store |
| ② 검색(Retrieval) | 사용자 질문과 관련된 청크 찾기 | Retriever, 유사도 검색 |
| ③ 생성(Generation) | 검색 문서와 질문을 LLM에 전달하여 답변 | Prompt, LLM, Output Parser |

```text
[사전 준비] 문서 → Chunking → Embedding → Vector DB
                                        │
[질문 시] 사용자 질문 → Retriever ────────┘
                         ↓ 관련 문서
                    Prompt(문서 + 질문)
                         ↓
                        LLM
                         ↓
                        답변
```

**RAG vs Fine-tuning:** RAG는 모델 가중치를 바꾸지 않고 관련 정보를 프롬프트에 제공한다. Fine-tuning은 모델 파라미터를 추가 학습한다. 최신 지식 검색과 문서 질의응답에는 RAG가 유리한 경우가 많다.



## 2. LCEL로 기본 RAG 체인 만들기

**LCEL(LangChain Expression Language)** 은 `|` 기호로 처리 단계를 연결하는 방식이다.

```text
질문 → {context: 문서 검색 및 포맷팅, question: 원본 질문}
     → Prompt → LLM → StrOutputParser → 답변
```

### 핵심 구성 요소

| 구성 요소 | 역할 |
|---|---|
| `retriever` | 질문에 관련된 `Document` 목록 반환 |
| `format_docs()` | `Document` 목록을 프롬프트에 넣을 문자열로 변환 |
| `RunnablePassthrough()` | 사용자 질문을 그대로 통과시킴 |
| `ChatPromptTemplate` | `{context}`, `{question}`을 합쳐 프롬프트 생성 |
| `ChatOpenAI` | 답변 생성 |
| `StrOutputParser()` | 모델 응답에서 텍스트 추출 |

### 예제 코드

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 문서 준비
Docs = [
    Document(page_content="RAG는 외부 문서를 검색해 답변을 생성하는 방식입니다.", metadata={"source": "rag"}),
    Document(page_content="LangChain은 LLM 앱 개발용 프레임워크입니다.", metadata={"source": "langchain"}),
]

# 2. 벡터 저장소와 검색기
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(Docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. 검색 문서 → 문자열

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 자료만 근거로 답하세요. 모르면 모른다고 하세요.\n\n{context}"),
    ("human", "{question}")
])

# 5. 체인 구성 및 실행
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-5-mini")
    | StrOutputParser()
)

print(rag_chain.invoke("RAG가 뭐야?"))
```

> 실행 전 관련 패키지(`langchain-core`, `langchain-openai`, `langchain-community`, `faiss-cpu`)와 OpenAI API 키 설정이 필요하다. 코드는 설명용 최소 예시다.

### 반드시 이해할 포인트

1. **질문 하나가 두 경로로 분기**된다. `context` 경로는 검색하고, `question` 경로는 원본 질문을 유지한다.
2. Retriever 출력은 **문자열이 아닌 `Document` 객체 리스트**다. 따라서 `format_docs()`가 필요하다.
3. `{context}`는 검색된 근거, `{question}`은 사용자가 실제로 묻는 내용이다.
4. `rag_chain.invoke(질문)` 한 번으로 검색 → 프롬프트 구성 → 생성이 실행된다.

**출처까지 포함하려면:**

```python
def format_docs(docs):
    return "\n\n".join(
        f"[출처: {doc.metadata.get('source', 'unknown')}] {doc.page_content}"
        for doc in docs
    )
```



## 3. RAG 프롬프트 엔지니어링

검색 결과를 넣기만 해서는 충분하지 않다. **문서에 근거할 것**, **근거가 없으면 모른다고 할 것**, **출처를 인용할 것** 등을 명시한다.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """다음 문서를 근거로 답변하세요.
1. 문서에 없는 사실을 만들어내지 마세요.
2. 근거가 부족하면 '자료에서 확인할 수 없습니다'라고 답하세요.
3. 가능한 경우 [문서 N] 형식으로 출처를 인용하세요.

문서:
{context}"""),
    ("human", "{question}")
])
```

- **기본 프롬프트:** 검색 문서 참고만 지시.
- **환각 억제 프롬프트:** 문서에 없는 정보는 답하지 않도록 지시.
- **출처 인용 프롬프트:** 검색된 문서 번호와 원문 출처 표시.
- **주의:** 프롬프트 지시만으로 근거의 정확성이나 올바른 인용이 보장되지는 않는다.



## 4. 고급 Retriever

### 4-1. MultiQueryRetriever — 질문을 여러 개로 바꿔 검색

사용자 질문 하나를 LLM이 여러 검색 질의로 바꿔, 각각 검색하고 결과를 통합한다.

```text
"이미지 분류 방법은?"
  ├─ "CNN 기반 이미지 분류 기법"
  ├─ "이미지 인식 신경망 구조"
  └─ "딥러닝 컴퓨터비전 분류 모델"
              ↓
      여러 번 검색 후 결과 통합
```

- **장점:** 원래 질문만으로는 놓치는 문서를 찾을 수 있다.
- **단점:** 추가 LLM 호출과 여러 번의 검색 때문에 지연·비용이 커질 수 있다.

### 4-2. ContextualCompressionRetriever — 검색 내용 압축

처음 검색한 문서에서 **질문에 필요하지 않은 부분을 필터링 또는 추출**하여 LLM에 전달한다.

- **장점:** 불필요한 문맥과 토큰 사용량 감소 가능.
- **단점:** 중요한 맥락까지 제거될 위험. 압축 과정 자체의 비용도 고려해야 한다.

| 비교 | MultiQuery | Contextual Compression |
|---|---|---|
| 개선 대상 | 검색 누락 | 검색 문서의 노이즈 |
| 핵심 방법 | 질의 확장 | 문서 내용 정제 |

> 고급 Retriever의 import 경로는 LangChain 버전에 따라 변경될 수 있으므로 실제 구현 시 사용하는 버전의 문서를 확인한다.



## 5. 대화형 RAG(Conversational RAG)

일반 RAG는 현재 질문만 검색하기 때문에 **이전 대화를 지칭하는 표현**을 이해하기 어렵다.

```text
사용자: "LangChain이 뭐야?"
AI: "LLM 앱 개발 프레임워크야."
사용자: "그걸로 RAG는 어떻게 만들어?"
```

여기서 `그걸로`가 무엇인지 검색기에 전달하려면 **대화 맥락을 반영해 독립적인 질문으로 재작성**해야 한다.

```text
이전 대화 + 현재 질문
          ↓
   질문 재작성 LLM
          ↓
"LangChain으로 RAG를 어떻게 만들어?"
          ↓
   Retriever → 답변 생성
```

### 주요 구성 요소

- `MessagesPlaceholder("chat_history")`: 프롬프트에 이전 대화 메시지 목록 삽입.
- **History-Aware Retriever:** 이전 대화를 이용해 검색용 질문을 재작성한 뒤 검색.
- `RunnableBranch`: 대화 기록이 있으면 질문 재작성, 없으면 원 질문 사용 등 조건 분기 가능.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "이전 대화를 참고해 현재 질문을 독립적인 검색 질문으로 재작성하세요."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
```

**핵심:** 검색용 질문을 재작성하는 것과 대화 내용을 장기 저장하는 것은 다른 작업이다. 이전 메시지 보관은 별도로 설계해야 한다.



## 6. RAG 평가와 품질 개선

**검색이 잘못되었는지, 생성이 잘못되었는지를 분리**하여 평가하는 것이 중요하다.

| 평가 지표 | 질문 | 단계 |
|---|---|---|
| Context Relevance | 검색 문서가 질문에 관련 있는가? | 검색 |
| Context Recall | 정답에 필요한 정보를 충분히 찾았는가? | 검색 |
| Faithfulness | 답변이 제공된 문서에 의해 뒷받침되는가? | 생성 |
| Answer Relevance | 답변이 사용자의 질문에 적절한가? | 생성 |

예) 문서에 마감일이 11월 10일이라고 되어 있는데 AI가 15일이라고 답하면, **검색은 성공했더라도 Faithfulness에 문제가 있는 것**이다.

### 자주 쓰는 개선 방법

| 증상 | 개선 아이디어 |
|---|---|
| 관련 문서가 검색되지 않음 | 청크 크기·임베딩 모델 변경, MultiQuery |
| 검색 결과가 너무 비슷함 | MMR, 검색 다양성 조정 |
| 검색 문서가 길고 산만함 | Contextual Compression |
| 문서 순위가 부정확함 | Reranking(재순위화) |
| 답변이 근거에 없는 내용을 포함함 | 프롬프트 제약, 출처 검사, 답변 검증 |
| 응답이 너무 느림 | 검색 횟수 조절, 캐싱, 모델·인덱스 최적화 |

### Reranking(리랭킹)

1차 검색으로 문서 20개를 찾은 다음, 질문과의 관련성을 **보다 정밀하게 재평가**해 상위 3개만 LLM에 제공하는 방법이다. 검색 결과의 품질을 높일 수 있지만 추가 계산 비용이 발생한다.

### Chunk Size와 Chunk Overlap

- **작은 청크:** 세밀하게 찾기 쉽지만 주변 맥락을 잃기 쉽다.
- **큰 청크:** 문맥이 풍부하지만 무관한 내용과 토큰이 증가한다.
- **Overlap:** 인접 청크가 일부 내용을 겹치게 하여 경계에서 문맥이 끊기는 문제를 완화한다.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_text(text)
```

> 최적 청크 크기·검색 개수(`k`)·점수 임계값은 문서 성격과 실제 질문 데이터로 실험해 정한다.


