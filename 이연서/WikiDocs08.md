# 위키독스 08

개념: RAG

## **1. RAG의 개념 & 기본 3단계**

`RAG(Retrieval-Augmented Generation)`: 외부 자료에서 관련 정보를 검색하고, 그 원문을 LLM의 입력 문맥에 추가하여 답변을 생성하는 방식

1. `인덱싱(Indexing)` : 문서를 분할하고 임베딩하여 검색 가능하게 준비하는 단계
2. `검색(Retrieval)` : 사용자 질문과 관련된 문서 조각 검색
3. `생성(Generation)` : 검색된 원문과 질문을 바탕으로 답변 생성
- 문서와 임베딩 설정이 유지된다면 질문마다 인덱싱을 다시 수행할 필요 X
- 기본적인 RAG에서는 질문마다 검색을 수행하고, 검색 결과를 참고하여 답변 생성
- `그라운딩(Grounding)`: 답변을 실제 자료에 근거하도록 하는 것
- 현재 구성하는 RAG는 LLM의 가중치를 변경하지 않고 검색 결과를 입력에 추가함
- RAG는 할루시네이션을 줄이는 데 도움을 주지만, 잘못된 검색이나 해석으로 오류가 발생할 수 있음

### 1-1) **RAG와 파인튜닝의 차이**

| 구분 | 기본 RAG | 파인튜닝 |
| --- | --- | --- |
| 방법 | 관련 자료를 검색하여 입력 문맥에 추가 | 특정 데이터로 모델을 추가 학습 |
| 파라미터 업데이트 | 채팅 LLM의 파라미터를 변경할 필요 없음 | 학습 대상 파라미터를 업데이트 |
| 활용 예 | 변경된 안내문이나 내부 문서를 참고하여 답변 | 특정 작업·응답 형식·문체 등에 맞게 모델 조정 |
- 기본 RAG: 안내문이 변경되어도 채팅 LLM을 다시 학습할 필요 X
- 파인튜닝: 사전 학습된 모델을 추가로 학습하는 것이며, 방식에 따라 전체 또는 일부/추가 파라미터를 학습해야 함
- RAG와 파인튜닝은 함께 사용할 수 있다
    - 파인튜닝한 모델에 검색된 자료를 제공하는 방식도 가능

## 2. **출처를 포함하는 문서 포맷팅**

```python
def format_docs_with_sources(documents):
    blocks = []

    for number, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "출처 미상")
        blocks.append(
            f"[문서 {number}] 출처: {source}\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(blocks)
```

- 기본 `format_docs`는 원문만 추출하므로 메타데이터에 있는 출처가 전달되지 않는다.
- 위 함수는 문서 번호·출처·원문을 하나의 문자열로 정리한다.
- 이미 검색된 문서를 LLM에 전달할 형태로 가공하며, 검색할 문서를 선택하지는 않는다.

체인에서는 다음과 같이 연결한다.

```
"context": retriever | format_docs_with_sources
```

## 3. **RAG 프롬프트 설계**

RAG 프롬프트에는 참고 자료뿐 아니라, 자료를 어떻게 사용하여 답할지도 명시해야 함

| 지시 | 목적 |
| --- | --- |
| 제공된 문서를 근거로 답변 | 답변 근거의 범위 설정 |
| 문서에 답이 없으면 확인할 수 없다고 안내 | 근거 없는 추측을 줄임 |
| 근거 문서 번호 표시 | 사용자가 답변의 근거를 확인하도록 도움 |

```python
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """제공된 문서를 근거로 답하세요.
문서에 답이 없으면 확인할 수 없다고 말하세요.
근거 문서 번호를 [문서 1]처럼 표시하세요.

참고 문서:
{context}"""
    ),
    ("human", "{question}"),
])
```

- `format_docs_with_sources`는 문서 번호·출처·원문을 입력에 포함하는 파이썬 함수
- 프롬프트에서 그 자료를 근거로 답하고, 근거 번호를 표시하도록 지시함
- 출처를 제공하거나 인용을 요청했다고 정확성이 자동으로 보장되지는 않음
    
    → 실제 문서가 해당 답변을 뒷받침하는지 확인이 필요
    

## 4. LangChain 내장 **RAG 체인 생성 헬퍼 함수**

- `create_stuff_documents_chain()` : 전달받은 문서들을 포맷팅하여 프롬프트에 넣고 답변을 생성하는 체인 구성
- `create_retrieval_chain()` : Retriever와 문서 답변 체인을 연결

```python
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

# model, retriever, helper_prompt는 미리 준비
# helper_prompt는 {context}와 {input}을 사용
document_chain = create_stuff_documents_chain(
    model,
    helper_prompt,
)

helper_rag_chain = create_retrieval_chain(
    retriever,
    document_chain,
)

result = helper_rag_chain.invoke({
    "input": "발표 시간은 몇 분인가요?"
})

print(result["answer"])   # 답변 문자열
print(result["context"])  # 검색된 Document 리스트
```

- 일반적인 벡터 스토어 `Retriever`를 연결하면 `"input"` 값이 검색 질문으로 전달됨
- `create_stuff_documents_chain()` 자체는 문서 검색 수행 X
- 기본 문서 포맷은 원문을 사용하며, 출처를 포함하려면 별도 포맷 설정이 필요
    - `format_docs_with_sources` 같은 함수 사용
- 최종 반환값은 답변과 검색 문서 등을 포함하는 딕셔너리

## 5. **MultiQueryRetriever**

### **5-1) 개념**

1. 사용자 질문 하나를 LLM으로 여러 검색용 질문으로 변형
2. 각 질문을 기존 Retriever에 전달하여 검색
3. 검색 결과를 합친 뒤 중복 문서를 제거

⇒ 따라서 질문 표현에 따라 관련 문서를 놓치는 것을 줄이는 데 도움을 줌

- 내부 LLM은 **검색용 질문 생성** 담당 → 최종 답변을 생성하는 역할 X
- 반환값: 중복을 제거한 `Document` 리스트

### 5-2) **활용 코드**

```python
from langchain_classic.retrievers.multi_query import (
    MultiQueryRetriever,
)

base_retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=model,
    include_original=True,
)

documents = multi_retriever.invoke(question)
```

**MultiQueryRetriever의 설정**

- `retriever` : 각 질문으로 실제 검색을 수행할 Retriever
- `llm` : 검색용 변형 질문을 생성할 모델
- `include_original=True` : 원래 사용자 질문도 검색 목록에 추가

**참고 사항**

- 기본 프롬프트는 변형 질문 3개를 생성하도록 요청한다.
- `k`는 **각 질문에서 검색할 최대 문서 수**
- 변형 질문 3개에 원래 질문까지 포함하고 `k=2`라면, 중복 제거 전 문서가 최대 8개가 모인다.
- `include_original=False`여도 변형 질문들로는 검색 수행

## 6. **ContextualCompressionRetriever**

### **6-1) 개념**

검색된 문서를 질문에 맞게 정제하여, 이후 LLM에 전달할 문맥을 추리는 방식

```
질문 → 기본 Retriever로 검색 → 압축기로 정제 → Document 리스트
```

**구성 요소**

- `base_retriever` : 먼저 문서를 검색
- `base_compressor` : 검색 결과를 질문에 맞게 정제

**압축기 2가지**

- `LLMChainExtractor` : LLM으로 문서에서 질문과 관련된 부분을 추출
    
    → 사용 모델: 채팅 LLM
    
- `EmbeddingsFilter` : 임베딩 유사도를 기준으로 문서 단위로 선택
    
    → 사용 모델: 임베딩 모델
    

**참고 사항**

- 목적: 일반적인 전체 요약보다 **현재 질문에 필요한 정보를 남기는 것**
- `ContextualCompressionRetriever` 를 통해 기본 검색기와 압축기를 연결
- 실제 정제 방식은 연결한 압축기에 따라 달라짐 (압축기 2가지 방식 참고)

### 6-2) 활용 코드

```python
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
)

question = "발표 시간은 몇 분인가요?"

# 공통: 먼저 최대 4개 문서 검색
base_retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

# 방법 1: LLM으로 관련 부분 추출
extractor = LLMChainExtractor.from_llm(model)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=extractor,
)

compressed_docs = compression_retriever.invoke(question)

# 방법 2: 임베딩 유사도로 문서 선택
embedding_filter = EmbeddingsFilter(
    embeddings=embeddings,
    k=None,
    similarity_threshold=0.75,
)

filtered_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=embedding_filter,
)

filtered_docs = filtered_retriever.invoke(question)
```

**주요 설정과 반환값**

- 기본 Retriever의 `k=4`: 먼저 검색할 최대 문서 수
- 필터의 `k=None`: 압축기에서 상위 개수 제한을 두지 않음
- `similarity_threshold=0.75`: 유사도가 0.75를 넘는 문서만 선택하는 예시 설정
- 반환값은 **정제된 `Document` 리스트**이며, 최종 답변 문자열 X
- 이후 `format_docs → prompt → model → 출력 파서`로 연결하여 답변 생성
- 벡터 스토어의 원본은 유지되고, 정제된 문서가 이번 검색 결과의 반환값
- 문맥을 줄일 수 있지만 추출용 LLM 호출이나 임베딩 계산이 추가되므로, 총비용이 항상 줄어드는 것은 아님!!

## 7. **대화형 RAG와 히스토리 인식 검색**

### 7-1) 개념

후속 질문에는 ‘그것’, ‘그걸로’처럼 이전 대화가 있어야 이해할 수 있는 표현이 포함될 수 있다

⇒ 히스토리 인식 검색을 통해 이전 대화와 현재 질문을 이용해 **독립적으로 이해할 수 있는 검색 질문**을 생성하여 해결!

```
이전 대화 + 현재 질문
          ↓
LLM으로 검색 질문 재작성
          ↓
Retriever에 전달하여 문서 검색
```

| 히스토리 | 처리 방식 |
| --- | --- |
| 없음 | 현재 질문을 그대로 검색 |
| 있음 | 대화 맥락을 반영해 질문을 재작성한 뒤 검색 |
- 재작성용 LLM은 최종 답변이 아니라 **검색용 질문**을 생성한다.
- 검색 단계의 반환값은 `Document` 리스트이다.
- 대화 기록은 프로그램에서 관리하고 입력으로 전달해야 한다.
- 답변 단계에만 히스토리를 제공하면 검색 단계에서 필요한 문서를 놓칠 수 있다.

### 7-2) **대화 기록과 MessagesPlaceholder**

```python
from langchain_core.messages import HumanMessage, AIMessage

chat_history = [
    HumanMessage(content="FAISS가 뭐야?"),
    AIMessage(content="벡터 유사도 검색을 위한 라이브러리야."),
]

input_data = {
    "input": "그걸 파일로 저장하려면?",
    "chat_history": chat_history,
}
```

- `chat_history`: 현재 질문 이전의 대화 메시지 리스트
- `input`: 현재 사용자의 질문
- `HumanMessage`: 사용자 메시지
- `AIMessage`: AI 메시지
- `MessagesPlaceholder(variable_name="chat_history")`: 대화 메시지를 역할과 순서를 유지하며 프롬프트에 삽입
- `rewrite_prompt.invoke(input_data)`: 메시지를 구성하는 단계이며
    - LLM 호출이나 질문 재작성까지 수행 X

### 7-3) **create_history_aware_retriever**

대화 히스토리 유무에 따라 검색 질문을 준비하고 문서를 검색하는 헬퍼 함수

```python
from langchain_classic.chains import (
    create_history_aware_retriever,
)

history_retriever = create_history_aware_retriever(
    llm=model,
    retriever=retriever,
    prompt=rewrite_prompt,
)

documents = history_retriever.invoke({
    "input": current_question,
    "chat_history": chat_history,
})
```

| 입력 상태 | 실행 흐름 |
| --- | --- |
| 히스토리 없음 | 현재 질문 → Retriever |
| 히스토리 있음 | **히스토리와 질문 → LLM으로 재작성** → Retriever |
- 반환값: 검색된 `Document` 리스트
- 재작성된 검색 질문 자체를 최종 결과로 반환하는 것 X
    - 최종 답변 생성 별도 LLM 호출이 필요
- 히스토리가 없으면 **질문** **재작성용** LLM 호출을 생략

### 7-4) **히스토리 인식 검색과 답변 생성 연결**

```python
answer_chain = create_stuff_documents_chain(
    model,
    answer_prompt,
)

conversational_rag = create_retrieval_chain(
    history_retriever,
    answer_chain,
)

result = conversational_rag.invoke({
    "input": current_question,
    "chat_history": chat_history,
})
```

- `history_retriever`: 필요하면 검색 질문을 재작성하고 문서를 검색
- `answer_chain`: 검색된 문서, 원래 질문, 이전 대화를 바탕으로 답변 생성
- 답변 프롬프트에도 `MessagesPlaceholder("chat_history")`를 넣어 대화 맥락을 전달
- 히스토리가 없어도 최종 답변 생성용 LLM은 실행됨

```
result["context"]  # 검색된 Document 리스트
result["answer"]   # 생성된 답변 문자열
```

### 7-5) **LCEL로 대화형 RAG 직접 구성하기**

미리 준비되어 있어야하는 요소들

- `rewrite_chain`: 히스토리와 질문으로 검색 질문을 재작성
- `retriever`: 문서 검색
- `format_docs`: 원문들을 문자열로 합치기
- `answer_prompt`, `model`: 답변용 프롬프트와 채팅 LLM

```python
from langchain_core.runnables import (
    RunnableBranch,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser

query_rewriter = RunnableBranch(
    (
        lambda x: len(x.get("chat_history", [])) > 0,
        rewrite_chain,
    ),
    lambda x: x["input"],
)

context_builder = (
    query_rewriter
    | retriever
    | format_docs
)

lcel_conversational_rag = (
    RunnablePassthrough.assign(context=context_builder)
    | answer_prompt
    | model
    | StrOutputParser()
)

answer = lcel_conversational_rag.invoke({
    "input": current_question,
    "chat_history": chat_history,
})
```

**구성 요소**

- `RunnableBranch` : 조건에 따라 실행 경로 선택
- `query_rewriter` : 검색에 사용할 질문 문자열 반환
- `context_builder` : 검색된 원문들을 합친 문자열 반환
- `RunnablePassthrough.assign()` : 기존 입력값을 유지하며 계산한 값을 추가

**참고 사항**

- 히스토리가 있으면 질문을 재작성하고, 없으면 현재 질문을 그대로 사용
- `assign()` 이후에는 `input`, `chat_history`, `context`가 답변 프롬프트에 전달됨
- `StrOutputParser()` 사용 시 최종 답변 문자열 반환

### 7-5) **대화 기록 갱신 및 다음 호출**

```python
from langchain_core.messages import HumanMessage, AIMessage

answer = lcel_conversational_rag.invoke({
    "input": current_question,
    "chat_history": chat_history,
})

# 현재 질문과 생성된 답변을 기록
chat_history.extend([
    HumanMessage(content=current_question),
    AIMessage(content=answer),
])
```

- `extend()`: 리스트에 메시지들을 추가하는 Python 메서드
    - 현재 코드 기준 대화 기록을 자동으로 history에 반영 X
    - 기록을 추가하는 것만으로 LLM을 호출하지는 않음
    - 다음 질문에서도 갱신된 `chat_history`를 입력으로 전달해야 한다.
- 헬퍼 체인의 반환값이 딕셔너리인 경우 AI 메시지에는 `result["answer"]`를 넣으면 됨

## 8. **RAG 평가의 두 축과 네 지표**

| 평가 대상 | 지표 | 확인하는 것 |
| --- | --- | --- |
| 검색 품질 | 컨텍스트 관련성 | 검색된 문서가 질문과 관련 있는가? |
| 검색 품질 | 컨텍스트 재현율 | 답변에 필요한 정보를 빠뜨리지 않고 가져왔는가? |
| 생성 품질 | 충실도 | 답변의 주장들이 검색된 문서로 뒷받침되는가? |
| 생성 품질 | 답변 관련성 | 사용자가 물어본 것에 적절히 답했는가? |
- 관련 있는 문서를 가져와도 필요한 정보가 빠지면 재현율이 낮을 수 있다
- 문서에 근거한 답변이어도 질문에 답하지 않으면 답변 관련성이 부족할 수 있다
- 질문의 주제에 맞는 답변이어도 문서에 없는 내용을 생성하면 충실도에 문제가 있다
- 충실도는 검색된 문서와의 일관성을 평가한다
    - 문서 자체가 사실인지와는 구분해야 한다

### 8-1) **RAG 평가 방법과 예제 코드의 한계**

평가에는 사용자 질문, 검색된 문서, 생성된 답변을 사용한다. 필요한 정보를 얼마나 가져왔는지 평가하려면 기준 정답이나 기준 문서도 필요하다.

**충실도의 기본 계산 관점**

![스크린샷 2026-11-03 오후 11.25.45.png](%EC%9C%84%ED%82%A4%EB%8F%85%EC%8A%A4%2008/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-11-03_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_11.25.45.png)

- 핵심 작업: 각 주장이 문서에서 뒷받침되는지 판단하는 것
    - 같은 단어가 많아도 수치/ 부정/ 주체/ 조건이 다르면 충실하지 않을 수 있기 때문
- 실제 평가에서는 사람이나 LLM 평가기/ 전용 모델 등을 활용
- 추가로 같은 테스트 질문 세트로 변경 전후 결과를 비교해야 개선 여부를 확인하기 쉬움

### 8-2) **RAG 품질 개선 전략**

| 문제 | 먼저 점검할 부분 | 개선 방법 |
| --- | --- | --- |
| 엉뚱한 문서 검색 | 검색·인덱싱 | 청크 분할, 임베딩 모델, 임계값·필터 점검 |
| 비슷한 내용 반복 | 검색 결과의 중복 | MMR |
| 필요한 정보 누락 | 원본 자료·검색 범위 | 색인 여부 확인, `k` 조정, MultiQuery |
| 문서와 다른 답변 | 생성 단계 | 근거 중심 프롬프트, 정보 부족 시 안내, 답변 검증 |
| 장황한 답변 | 출력 지시·입력 문맥 | 길이·형식 지정, 불필요한 문맥 압축 |
| 느린 응답 | 단계별 처리 시간 | 캐싱, 모델 경량화, 검색·생성 최적화 |
- `k`를 늘리면 필요한 정보를 더 찾을 수 있지만, 무관한 문서와 입력량도 늘어날 수 있다.
- 낮은 `temperature`가 사실 정확성을 보장하지는 않는다.
- 스트리밍은 답변 일부를 먼저 보여 주는 방식이며, 전체 완료 시간 단축을 보장하지는 않는다.
- 같은 테스트 질문으로 변경 전후를 비교하고, 설정을 단계적으로 바꿔 효과를 확인한다.

### 8-3) **청크 크기와 겹침 조정**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    length_function=len,
)

# text는 분할할 원문 문자열
chunks = splitter.split_text(text)
```

**설정**

- `chunk_size` : 청크 하나의 최대 크기
- `chunk_overlap` : 인접 청크 사이의 겹침 목표
- `length_function=len` : 크기를 문자 수로 계산

**참고 사항**

- 청크가 너무 작으면 설명이나 예외 조건이 나뉘어 문맥이 부족해질 수 있다.
- 청크가 너무 크면 여러 주제와 불필요한 내용이 섞일 수 있다.
- 겹침은 청크 경계에서 정보가 끊기는 문제를 줄이는 데 도움을 준다.
- 겹침을 늘리면 중복 내용, 청크 수, 저장량과 임베딩 작업이 증가할 수 있다.
- 실제 청크 길이와 겹침은 문단 등의 분할 경계에 따라 달라질 수 있다.
- 적절한 값은 실제 테스트 질문으로 검색 품질을 비교하여 정한다.

**청크 설정을 바꿀 시**

```
원문 다시 분할 → 새 청크 임베딩 → 검색 인덱스 갱신
```

추가로 이미 저장된 벡터가 설정 변경만으로 자동 갱신되지는 않는다.

## 9. 최종 정리

**1. 대화형 RAG**

이전 대화를 참고해 현재 질문을 독립적으로 이해할 수 있는 검색 질문으로 재작성한다.

```
히스토리 없음 → 현재 질문 그대로 검색
히스토리 있음 → LLM으로 검색 질문 재작성 → 검색
```

- 재작성용 LLM과 최종 답변 생성용 LLM의 역할을 구분한다.
- 히스토리가 없어도 최종 답변 생성은 수행한다.
- `MessagesPlaceholder`는 이전 메시지들을 역할과 순서를 유지하며 프롬프트에 삽입한다.
- 대화 기록은 프로그램에서 관리하고 다음 호출에 전달해야 한다.
- `chat_history.extend(...)`는 기록을 추가할 뿐, LLM을 호출하지 않는다.

**2. LCEL 구성**

```
입력: input + chat_history
          ↓
RunnableBranch로 검색 질문 결정
          ↓
Retriever → format_docs
          ↓
assign으로 기존 입력에 context 추가
          ↓
답변 프롬프트 → LLM → 출력 파서
```

- `RunnableBranch`: 조건에 따라 실행 경로 선택
- `RunnablePassthrough.assign()`: 기존 입력을 유지하며 계산한 값 추가

| 구성 방식 | 배운 예제의 반환값 |
| --- | --- |
| `create_retrieval_chain()` 헬퍼 | 딕셔너리 — `result["answer"]`, `result["context"]` |
| 출력 파서로 끝나는 직접 구성 체인 | 답변 문자열 |

**3. RAG 평가**

| 지표 | 핵심 질문 |
| --- | --- |
| 컨텍스트 관련성 | 검색 문서가 질문과 관련 있는가? |
| 컨텍스트 재현율 | 필요한 정보를 빠뜨리지 않고 가져왔는가? |
| 충실도 | 답변이 검색 문서로 뒷받침되는가? |
| 답변 관련성 | 사용자가 물은 것에 적절히 답했는가? |
- 문서에 근거해도 질문에 답하지 않으면 답변 관련성이 부족할 수 있다.
- 질문에 맞는 종류의 답변이어도 문서와 다르면 충실도에 문제가 있다.
- 단어 겹침만으로 의미의 일치나 충실도를 판단하기 어렵다.
- 같은 테스트 질문으로 변경 전후 결과를 비교한다.

**4. 개선과 재인덱싱**

- 필요한 자료 누락 → 색인 상태와 검색 범위 점검, `k`·MultiQuery 등 검토
- 정확한 문서를 가져왔는데 답변이 다름 → 생성 지시와 문서 활용·답변 검증 점검
- 청크가 너무 작음 → 문맥이나 조건이 분리될 수 있음
- 청크가 너무 큼 → 불필요한 내용이 섞일 수 있음
- 겹침 증가 → 경계의 정보 손실을 줄일 수 있지만 중복·저장량·임베딩 작업 증가

청크 설정을 바꾸면:

```
원문 다시 분할 → 새 청크 임베딩 → 검색 인덱스 갱신
```

**5. 운영 시 고려할 점**

- 긴 히스토리: 최근 대화 유지, 오래된 대화 요약, 토큰 수 제한
- 반복 계산: 임베딩 등의 결과를 캐싱하여 재사용
- 스트리밍: 답변 일부를 먼저 보여 주며, 전체 완료 시간 단축을 보장하지는 않음