# Ch.08 RAG(검색 증강 생성) 파이프라인 정리

## 학습 목표
- RAG 개념과 환각 해결 원리 이해
- LCEL로 기본 RAG 파이프라인 구성
- 고급 Retriever(MultiQuery, ContextualCompression) 활용
- 대화형 RAG(히스토리 인식) 구축
- RAG 품질 평가·개선 전략

---

## 1. RAG란?

- **환각(Hallucination)**: LLM이 학습 데이터에 없는 내용을 사실처럼 생성하는 현상
- **RAG(Retrieval-Augmented Generation)**: 답변 생성 **전에** 외부 지식을 검색해 컨텍스트로 함께 제공 → 정확도 ↑, 환각 ↓
- 비유: 클로즈드북 시험(기존 LLM) vs **오픈북 시험(RAG)**
- **그라운딩(Grounding)**: 답변을 실제 데이터·문서에 근거하게 하는 것
- **RAG vs 파인튜닝**: RAG는 모델을 수정하지 않고 프롬프트에 문서를 추가 → 빠르고 경제적. 파인튜닝은 가중치를 변경(비용·시간 ↑). 보통 RAG를 먼저 시도.

### RAG 3단계
| 단계 | 하는 일 | 대표 API |
| --- | --- | --- |
| 1. 인덱싱 | 문서 → 청크 → 임베딩 → 벡터 DB 저장 | `RecursiveCharacterTextSplitter`, `FAISS.from_documents()` |
| 2. 검색 | 질문과 유사한 청크 검색 | `as_retriever()`, `retriever.invoke()` |
| 3. 생성 | 검색 결과 + 질문을 LLM에 전달 | `ChatPromptTemplate`, `ChatOpenAI` |

---

## 2. 기본 RAG 체인 (LCEL)

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-5-mini", temperature=0)
    | StrOutputParser()
)
rag_chain.invoke("RAG가 무엇인가요?")
```

| 구성 요소 | 역할 |
| --- | --- |
| `retriever \| format_docs` | 질문으로 문서 검색 후 텍스트로 변환 |
| `RunnablePassthrough()` | 원본 질문을 그대로 전달 |
| `ChatPromptTemplate` | 문서 + 질문 결합 |
| `ChatOpenAI` | 답변 생성 |
| `StrOutputParser` | 문자열 추출 |

- 딕셔너리 구문 = `RunnableParallel` → 검색과 질문 전달을 동시에 처리

### format_docs
- Document 리스트 → 텍스트 변환 함수
- ❌ `str(docs)` 처럼 그대로 넣으면 metadata·대괄호까지 들어가 LLM이 혼란, 토큰 낭비
- ✅ `"\n\n".join(doc.page_content for doc in docs)`
- 실무: **출처 포함 버전**으로 `[문서 N - 출처: ..., p.N]` 형태로 만들어 LLM이 인용하게 함

### 헬퍼 함수 방식
- `create_stuff_documents_chain`: 검색된 문서를 모두 프롬프트에 넣고 답변 생성
- `create_retrieval_chain`: 검색기 + 문서 결합 체인 연결
- LCEL 직접 구성 = 유연함 / 헬퍼 = 간결함 (단, 일부 레거시 체인은 deprecated)

---

## 3. RAG 프롬프트 엔지니어링

| 요소 | 예시 |
| --- | --- |
| 역할 부여 | "당신은 기술 문서 전문가입니다" |
| 컨텍스트 주입 | `{context}` 변수 |
| 답변 제약 | 문서에 없으면 "해당 정보가 문서에 없습니다" |
| 출처 인용 | `[문서 N]` 형태로 인용 |
| 형식 지정 | "3문장 이내로 답변" |

- **가장 중요한 원칙: "모르면 모른다고 하게 만들기"**
- 규칙은 번호를 매겨 나열하면 LLM이 더 잘 따름
- 프롬프트가 너무 길면 컨텍스트 윈도우 낭비 → 핵심만

---

## 4. 고급 Retriever

### MultiQueryRetriever
- LLM이 원본 질문을 **3~5개 변형 질문**으로 재작성 → 각각 검색 → 합집합(중복 제거)
- 효과: 표현이 달라 놓치던 문서까지 검색 (검색 범위 확장)
- 비용: 변형 생성용 **LLM 호출 추가**

```python
MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=ChatOpenAI(model="gpt-5-mini", temperature=0),
)
```

### ContextualCompressionRetriever
- 검색된 문서에서 **질문과 관련된 부분만 추출**, 무관한 문서는 제거 → 컨텍스트 정확도 ↑, 토큰 비용 ↓

| 압축기 | 방식 | 특징 |
| --- | --- | --- |
| `LLMChainExtractor` | LLM으로 관련 부분 추출 | 정확, LLM 비용 추가 |
| `EmbeddingsFilter` | 임베딩 유사도로 문서 필터링 | 저렴, 문서 단위 제거만 가능 |

```python
ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(llm),
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
)
```

### 기타
- **EnsembleRetriever**: 벡터 검색 + 키워드 검색 가중 결합
- 조합 전략: MultiQuery로 폭넓게 검색 → Compression으로 핵심만 추출 (단, 레이턴시·비용 증가 주의)

---

## 5. 대화형 RAG (히스토리 인식)

- 문제: "그걸로 뭘 할 수 있어?" 같은 **대명사·문맥 의존 질문**은 단독으로 검색하면 실패
- 해결: 히스토리 + 현재 질문 → LLM이 **독립적인 검색 쿼리로 재작성** → 검색
  - 예: "그걸로 RAG를 어떻게 만들어?" → "LangChain으로 RAG를 어떻게 만드나요?"
- 히스토리가 없으면 원본 질문 그대로 검색

### LCEL 구현 구조
```python
query_rewriter = RunnableBranch(
    (lambda x: len(x.get("chat_history", [])) > 0, rewrite_chain),  # 히스토리 있음 → 재작성
    lambda x: x["input"],                                           # 없음 → 원본 그대로
)

rag_chain = (
    RunnablePassthrough.assign(context=RunnableLambda(build_context))
    | answer_prompt   # MessagesPlaceholder("chat_history") 포함
    | llm
    | StrOutputParser()
)
rag_chain.invoke({"input": "...", "chat_history": [HumanMessage(...), AIMessage(...)]})
```

- `RunnablePassthrough.assign()`: 원본 입력에 새 키(`context`)를 추가
- `create_history_aware_retriever` + `create_retrieval_chain` 조합이 있으며, 레거시 `ConversationalRetrievalChain`은 deprecated
- 히스토리가 길어지면 토큰 비용 급증 → **윈도우(최근 5~10턴) / 요약 / trim_messages** 등 메모리 전략 적용 (Ch.09)

---

## 6. RAG 평가와 품질 개선

### 평가 지표
| 구분 | 지표 | 의미 | 낮으면 |
| --- | --- | --- | --- |
| 검색 | Context Relevance | 검색 문서가 질문과 관련 있는가 | 엉뚱한 문서 검색 |
| 검색 | Context Recall | 필요한 정보가 모두 검색됐는가 | 정보 누락 |
| 생성 | **Faithfulness(충실도)** | 답변이 검색 문서에 근거하는가 (가장 중요) | 환각 |
| 생성 | Answer Relevance | 답변이 질문에 적절한가 | 동문서답 |

- 실무 도구: **RAGAS**, **LangSmith evaluate()** (예제의 키워드 기반 평가는 데모용)
- 핵심: 문제가 **검색 단계인지 생성 단계인지 구분**해야 올바른 해결책을 찾는다.

### 문제별 개선 전략
| 문제 | 원인 | 개선 전략 |
| --- | --- | --- |
| 엉뚱한 문서 검색 | 검색 품질 ↓ | chunk_size/overlap 조정, 임베딩 모델 변경, MMR |
| 필요한 정보 누락 | 검색 범위 부족 | k 증가, MultiQueryRetriever |
| 환각 발생 | 충실도 ↓ | 프롬프트 강화, temperature ↓ |
| 답변이 장황함 | 불필요한 컨텍스트 | ContextualCompression, chunk_size ↓ |
| 응답 느림 | 검색/생성 병목 | 캐싱, 경량 임베딩, 스트리밍, k ↓ |

### 리랭킹(Reranking)
- 1차 검색(벡터 유사도, 빠름·대략적) → 2차 **Cross-Encoder**로 정밀 재정렬(느리지만 정확)
- 일반적으로 후보 20~50개 → 상위 3~5개 선별
- LangChain: `CohereRerank`, `CrossEncoderReranker`를 `ContextualCompressionRetriever`의 `base_compressor`로 사용

### chunk_size 최적화
| chunk_size | 장점 | 단점 | 적합 상황 |
| --- | --- | --- | --- |
| 200~500 | 정확, 노이즈 적음 | 맥락 부족 | FAQ, 정의 질문 |
| 500~1000 | 균형 | - | 일반 문서 QA |
| 1000~2000 | 풍부한 맥락 | 노이즈·비용 ↑ | 복잡한 주제, 분석 |

- 가장 큰 영향을 주는 파라미터. 출발점은 **500~1000자**, `chunk_overlap`은 chunk_size의 **10~20%**
- 테스트 질문 세트를 만들어 값별 평가 점수를 비교하는 것이 가장 좋음

---

## 7. 핵심 API 정리
| API | 용도 |
| --- | --- |
| `RunnablePassthrough()` | 입력을 그대로 전달 |
| `RunnablePassthrough.assign(**kwargs)` | 원본 입력에 키-값 추가 |
| `MultiQueryRetriever.from_llm(retriever, llm)` | 변형 쿼리로 검색 확장 |
| `ContextualCompressionRetriever(base_compressor, base_retriever)` | 관련 부분만 압축 반환 |
| `create_history_aware_retriever(llm, retriever, prompt)` | 히스토리 기반 쿼리 재작성 검색기 |
| `create_stuff_documents_chain(llm, prompt)` | 문서를 프롬프트에 넣어 답변 생성 |
| `create_retrieval_chain(retriever, combine_docs_chain)` | 검색 → 결합 → 생성 전체 체인 |
| `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)` | 문서 분할 |