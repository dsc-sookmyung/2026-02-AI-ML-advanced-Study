# Ch.07 임베딩과 벡터 스토어 정리

> 참고: https://wikidocs.net/333791

## 왜 필요한가?

'개'와 '반려견'은 문자 자체는 다르지만 의미가 비슷하다. **임베딩(Embedding)**은 텍스트를 숫자 벡터로 변환해 의미 기반 검색을 가능하게 한다.

- **코사인 유사도**: 두 벡터 방향의 유사성을 측정한다. `1`은 같은 방향, `0`은 직교, `-1`은 반대 방향을 뜻한다.
- **벡터 스토어(Vector Store)**: 임베딩 벡터·원문·메타데이터 등을 저장하고 가까운 벡터를 검색한다.
- **ANN**: 모든 벡터를 완전 탐색하지 않고 가까운 후보를 효율적으로 근사 검색한다.

## 임베딩 모델

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
question_vector = embeddings.embed_query("RAG란 무엇인가요?")
document_vectors = embeddings.embed_documents(["RAG는 외부 문서를 검색해 답변합니다."])
```

- `embed_query()`는 질문 하나, `embed_documents()`는 여러 문서를 벡터화한다.
- OpenAI 임베딩은 API 기반, Hugging Face 임베딩은 로컬 모델 실행이 가능하다(모델에 따라 성능·환경 차이).

## 벡터 검색 예시

```python
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

docs = [Document(page_content="RAG는 검색 증강 생성입니다.", metadata={"source": "rag.pdf"})]
vectorstore = FAISS.from_documents(docs, embeddings)
results = vectorstore.similarity_search("RAG가 뭐야?", k=1)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
found_docs = retriever.invoke("RAG란?")
```

- **FAISS**: 효율적인 벡터 검색 라이브러리로 프로토타입에 유용.
- **Chroma**: 영속 저장과 메타데이터 조건 검색을 지원하는 벡터 DB.
- **Retriever**: 질문을 받아 관련 문서들을 반환하는 표준 인터페이스. 벡터 스토어 위에 구성할 수 있다.

## 검색 전략

| 전략 | 핵심 | 주의점 |
|---|---|---|
| Similarity | 가장 유사한 문서 `k`개 | 결과끼리 중복 가능 |
| MMR | 질문 유사도 + 결과 다양성 | `fetch_k`, `lambda_mult` 조절 |
| Score Threshold | 임계 점수 이상만 반환 | 결과가 0개일 수도 있음 |

## 전체 흐름과 연결

`문서 → 임베딩 → 벡터 스토어에 저장` / `질문 → 임베딩 → 유사 문서 검색 → RAG의 LLM에 근거 제공`

**핵심:** 임베딩·검색 자체는 답변을 생성하지 않는다. 검색한 내용을 LLM에 넣는 단계까지 연결하면 RAG가 된다.
