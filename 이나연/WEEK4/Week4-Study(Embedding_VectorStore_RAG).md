# Week 3 - Study (AI Agent 및 Tool Calling)

### 📒 강의 개요

7장 - 임베딩과 벡터스토어

[https://wikidocs.net/333791](https://wikidocs.net/333791)

8장 - RAG 파이프라인
[https://wikidocs.net/333792](https://wikidocs.net/333792)


## Cp. 07, 08 - 임베딩과 벡터스토어 / RAG

### 핵심 개념 및 용어 정리

---

들어가기 전에, 6강 간단 정리하기

Cp.06 문서 로더와 텍스트 분할

- Document 객체와 메타데이터
    - LangChain의 Document(문서) 객체 : 텍스트 내용과 메타데이터를 하나로 묶어 관리하는 표준 자료 구조 (하나의 통일된 형식, 표준 자료 구조)
        - 텍스트 내용과 메타데이터 하나로 묶어 관리
    - Document 객체의 두 가지 핵심 속성
        - page_content - 문서의 실제 텍스트 내용 (문자열)
        - metadata - 출처, 페이지 번호, 작성일 등 부가 정보 (딕셔너리)
        - 참고로 Document 객체는 Python의 일반 클래스 인스턴스
- Document Loader (문서 로더)
    - 어떤 형식의 파일이든 .load() 메서드 ⇒ Document 객체 리스트로 변환
        - 로더의 출력이 동일한 List[Document] 형식
    - 다양한 형식의 데이터 소스에서 텍스트를 추출하여 Document 객체로 변환
    - 공통 메서드
        - .load()
        - .lazy_load()
    - 주요 문서 로더 (by. LangChain)
        - TextLoader
        - PyPDFLoader
        - CSVLoader
        - WebBaseLoader
        - UnstructuredFileLoader
    - 인코딩 지정 주의
- Text Splitter (텍스트 분할기)
    - 전체 텍스트 한 번에 LLM에게 다 보낼 수 X
        - 왜냐하면 LLM은 Context Window라는 한 번에 처리 가능한 토큰 수의 한계가 존재하기 대문.
        - 또한 RAG 시스템에서는 질문과 관련된 부분만 찾아서 전달하기 때문에
        - 따라서 문서를 의미있는 조각 chunk로 나누는 것이 필수
    - Text Splitter : 대용량 문서를 LLM이 처리하기 적합한 크기의 작은 청크(chunk)로 나누는 컴포넌트
    - 두 가지 핵심 파라미터
        - chunk_size - 각 청크의 최대 크기 (문자 수)
        - chunk_overlap - 인접한 청크 간 겹치는 부분의 크기 (경계에 있는 정보도 완전한 문맥 유지 가능)
    - 다양한 텍스트 분할 전략 중에서는 RecursiveCharacterTextSplitter를 가장 많이 씀
        - 먼저 의미 있는 경계(단락, 줄바꿈, 문장)를 찾아 그 지점에서 분할함. "재귀적"이라는 이름이 붙은 이유는 큰 단위부터 작은 단위까지 순서대로 시도하기 때문.
        - 핵심은 항상 가장 큰 의미 단위부터 시도한다는 것.
        - RecursiveCharacterTextSplitter는 분리자 계층(`\n\n` -> `\n` -> -> `""`)을 재귀적으로 적용하여 의미 단위를 최대한 보존하면서 분할한다
    - RAG 파이프라인에서는 **반드시 `split_documents()`를 사용**
        - 메타 데이터를 자동 상속하기 때문
    - 커스텀 분리자 직접 지정 가능

- 코드 분할기와 특수 분할기
    - 코드 분할기는 프로그래밍 언어의 문법 구조를 이해.
    - Python의 경우 클래스, 함수, if/for 블록 등의 경계에서 분할하여 논리적으로 완전한 코드 블록을 유지
    - 소스코드를 RAG에 활용할 때 사용
    - MarkdownHeaderTextSplitter
        - 마크다운 제목(`#`, `##`, `###`)을 기준으로 문서를 분할. 일반 분할기와 다른 핵심 차이점은 **제목이 메타데이터에 저장**
    - SemanticChunker(의미적 분할기)는 문자 수나 구분자가 아닌 **문장 간 의미적 유사도**를 기반으로 분할
        - 임베딩 모델을 사용하여 인접 문장의 의미가 크게 달라지는 지점을 경계로 인식함
        - 임베딩 모델 호출해야 해서 비용과 시간이 더 든다.. (검색 품질 매우 중요한 고급 RAG 시스템에서 사용)
    - 실무에서는 두 분할기 조합하여 사용하는 경우가 많음 (2단계 분할 전략)

### Ch.07 임베딩과 벡터 스토어

- 임베딩(Embedding)
    - 텍스트를 고차원 숫자 벡터(배열)로 변환
    - 의미의 유사도를 수학적으로 계산할 수 있게 만들어줌
    - 의미가 비슷한 텍스트는 벡터 공간에서 가까이 위치하고, 의미가 다른 텍스트는 멀리 위치
    - 임베딩 동작 과정
        1. 텍스트 → 임베딩 모델 입력
        2. 모델이 텍스트 의미 분석
        3. 숫자 벡터 출력 (고차원 벡터)
        4. 벡터 간 거리 유사도 계산 (가까울수록 의미 비슷)
- 코사인 유사도 (Cosine Similarity)
    - 의미적 거리 측정
    - 두 벡터가 가리키는 방향이 얼마나 같은지 측정
    - 코사인 유사도 값이 1이면 완전히 같은 의미
- 정리) 임베딩은 텍스트를 고차원 숫자 벡터로 변환하여 의미적 유사도를 수학적으로 계산할 수 있게 해줌. 코사인 유사도를 사용하면 두 텍스트의 의미적 거리를 0~1 사이의 숫자로 표현할 수 있음.
- LangChain 임베딩 모델 : OpenAI, HuggingFace
    - 실제로 텍스트를 벡터로 변환하는 방법.
    - LangChain은 다양한 임베딩 모델 제공업체를 동일한 인터페이스로 사용 가능하게 해줌
    - 두 가지 핵심 메서드
        - embed_query(text) (문자열 1개 → 벡터 1개 / 검색 쿼리 임베딩용)
        - embed_documents(texts) (문자열 리스트 → 벡터 리스트 / 문서 청크 일괄 임베딩용)
    - OpenAIEmbeddings
        - text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002 (대부분 3-small로 많이 함)
    
    <aside>
    💡
    
    embed_query() : 사용자가 질문할 때
    
    embed_documents() : 문서를 벡터 스토어에 저장할 때
    
    </aside>
    
    - OpenAI 임베딩은 API 호출 비용 발생, HuggingFace의 오픈소스 임베딩 모델 로컬에서 실행해서 비용 줄일 수 있음
        - 실무에서는 프로토타입 단계에서 OpenAI 임베딩으로 빠르게 시작하고, 비용이나 보안 이슈가 생기면 HuggingFace 로컬 모델로 전환하는 전략이 일반적
    - 주의! 저장할 때와 검색할 때 다른 임베딩 모델 사용 X
        
        <aside>
        💡
        
        정리: LangChain은 `embed_query()`와 `embed_documents()`라는 통합 인터페이스로 다양한 임베딩 모델을 지원. OpenAI는 클라우드 기반으로 빠르게 시작할 수 있고, HuggingFace는 로컬에서 무료로 실행할 수 있음. 저장과 검색에 반드시 같은 임베딩 모델을 사용해야함.
        
        </aside>
        
- Vector Store의 원리와 종류
    - Vector Store - 임베딩 벡터 저장하고, 쿼리 벡터와 가장 유사한 벡터 빠르게 찾아주는 특수한 DB
        - like 도서관, 의미적 검색 처리
    - 벡터 스토어 단계별 핵심 동작
        1. 인덱싱(Indexing) : 문서 청크들을 임베딩 벡터로 변환하여 저장. 원본 텍스트, 메타데이터, 벡터를 함께 보관.
        2. 검색(Search) : <u>사용자 질문(쿼리)을 임베딩 벡터로 변환</u>한 뒤, 저장된 벡터들과 유사도를 비교하여 가장 가까운 K개를 반환
        3. 결과 반환 : 유사도가 높은 순서대로 원본 텍스트(Document 객체)를 반환
    
    <aside>
    💡
    
    ANN(Approximate Nearest Neighbor, 근사 최근접 이웃) 검색이란 정확한 최근접 이웃 대신 **거의 정확한** 이웃을 훨씬 빠르게 찾는 알고리즘.
    
    모든 벡터 하나하나 비교하기보다는, 벡터 공간을 여러 구역으로 나눠서 해당 구역만 검색
    
    </aside>
    
    - LangChain 주요 벡터 스토어
        - FAISS (설치 간편, 빠름, 인메모리 고속 검색)
        - Chroma (영속성, 메타데이터 필터링 필요할 때)
        - Pinecone (관리형 서비스)
    - 벡터스토어는 임베딩 벡터 저장하고 유사도 기반 검색 수행하는 특수 DB
- FAISS - 인메모리 고속 벡터 검색
    1. 문서 준비 - Document 객체 리스트
    2. 벡터스토어 생성 - `FAISS.from_documents(docs, embeddings)`
    3. 유사도 검색 - `similarity_search(query, k=4)`
    4. 저장/로드 - `save_local()`과 `load_local()`로 디스크에 저장
    - 유사도 점수 확인 - 검색 결과의 신뢰도 판닪가능
        - 대신 점수가 높을수록 신뢰도 낮을 수 있음. (값이 작을수록 유사도 점수가 높음, 유사도 점수가 distance로 표현되기 때문)
        - 코사인 유사도와 반대 주의!!
    - FAISS는 기본적으로 메모리에만 존재하기 때문에, 프로그램 종료하면 데이터 사라짐.
        - save_local()과 load_loca() 사용하여 디스크에 저장했다가 다시 로드
    
    <aside>
    💡
    
    `allow_dangerous_deserialization=True` 파라미터가 왜 필요할까?
    ⇒ FAISS 인덱스를 로드할 때 내부적으로 Python의 pickle을 사용하는데, pickle은 악의적인 코드를 실행할 수 있는 보안 위험이 있다고 함. LangChain은 이 위험을 인지하고 있다는 것을 명시적으로 확인하기 위해 이 파라미터 요구.본인이 직접 저장한 인덱스 로드하는 경우 → true로 설정
    
    </aside>
    
- Chroma - 영속성O 벡터 DB
    - 왜 필요?
        - FAISS는 빠르지만 메타데이터 기반 필터링 지원 X
        - Chroma → 메타데이터 필터링 지원 O + 설치와 사용 간편
    - 기본 사용 흐름 FAISS와 비슷, `persist_directory`를 지정하면 데이터 자동으로 디스크에 저장
    - 메타데이터 필터링 : 벡터 유사도 검샥 + 메타데이터 조건 결합 ⇒ 더 정확한 결과
        - 필터연산자 사용
        - `$eq`, `$gt` 등의 연산자로 조건부 검색 가능
    - 주의 : Chroma에 동일 문서 중복 추가 주의하기. 매번 새로운 문서로 저장됨.

- Retriever 인터페이스와 검색 전략
    - 지금까지는 벡터스토어의 similarity_search() 메서드 직접 호출하여 검색
    - LCEL 체인 - Retriever 인터페이스 사용
        - LangChain 표준 검색 인터페이스, 벡터 스토어뿐아니라 어떤 검색 소스든 동일한 방식으로 체인 연결
    - Retriever란?
        - 자연어 쿼리 → 관련 Document 리스트 반환O 표준 인터페이스
        - `as_retriever()` 메서드 호출
    - `as_retriever()` 메서드 - `search_type` 파라미터
        - 검색 전략 3가지
        1. similarity : 코사인 유사도 상위 K개 반환
        2. mmr : 유사도 + 다양성 균형
            1. MMR, Maximal Marginal Relevance, 최대 한계 관련성
            2. 관련성은 높되 서로 다른 내용의 문서 골라줌
        3. similarity_score_threshold : 유사도 임계값 이상만 반환
    - Retriever LCEL 페인에 연결하기 (8장에서 더 자세히)
        - search_kwargs 설정 실수 조심하기

### Ch.08 **RAG(검색 증강 생성) 파이프라인**

- RAG - LLM 환각 잡는 핵심 패턴
    - 문제: Hallucination ⇒ 해결: RAG
    - RAG(Retrieval-Augmented Generation)
        - LLM이 답변 생성하기 전에 외부 지식 관련 정보를 먼저 검색,
        - 그 정보를 context로 함께 제공하여 답변의 정확도 높이는 패턴
        - 기존 LLM은 모든걸 외워서 답했다면 RAG는 관련 참고자료를 찾아보고 그걸 바탕으로 답변한다.
    - RAG의 세 단계
        1. 인덱싱(Indexing)
            1. 문서 chunk로 나누고 임베딩 → 벡터 DB 저장
        2. 검색(Retrieval)
            1. 사용자 질문(쿼리)과 유사한 문서 chunk → 벡터 DB에서 검색
        3. 생성(Generation)
            1. 검색된 문서와 질문 → LLM에 전달하여 답변 생성
    
    <aside>
    💡
    
    RAG vs 파인튜닝
    
    : RAG는 모델 자체 수정 X, 프롬프트에 검색된 문서 추가하는 방식
    
    파인튜닝은 모델의 가중치 직접 변경 → 비용과 시간 많이 든다. 
    
    * 그라운딩 : LLM의 답변ㅇㄹ 실제 데이터나 문서에 근거하여 생성하도록 하는 것, RAG가 대표적인 그라운딩 구현 방법
    
    </aside>
    
- 기본 RAG 체인 - LCEL로 구성
    - LCEL 활용하면 RAG 파이프라인 선언적이고 깔끔하게 구성 가능.
    - 구조) 사용자 질문 → {context: retriever, question: passthrough} → 프롬프트 → LLM → 파서 → 답변
    - RunnableParallel로 검색과 질문 전달을 동시에 처리하는게 핵심.
        - RunnablePassthrough - LCEL 컴포넌트, 입렭데이터를 아무런 변환 없이 그대로 다음 단계로 전달함.
    - (실습 중) format_docs함수
        - retriever가 반환하는 것은 Document 객체 리스트 → 일반 LLM은 일반 텍스트를 입력으로 받음
        - format_docs 함수가 이 변환 담당
        - 이 함수 없이 직접 프롬프트에 전달되면 LLM이 혼란스러워 하니 잘 챙기기
- RAG 프롬프트 엔지니어링
    - 명확한 지시가 프롬프트에 포함되어야 하며, 프롬프트 어떻게 작성하느냐에 따라 RAG 답변 품질이 크게 달라짐
    - RAG 프롬프트 설계할 때 고려해야 할 핵심 요소들
        1. 역할 부여
        2. context 주입 (검색된 문서 삽입 위치 지정)
        3. 답변 제약 (문서에 없는 내용 답변 금지)
        4. 출처 인용
        5. 형식 지정
    - RAG 프롬프트에서 중요한 원칙은 모르면 모른다고 하게 만드는 것.
    - LangChain의 RAG 체인 헬퍼 함수 O
        - LCEL 직접 구성 - 유연 / 코드길어짐
        - 헬퍼 함수 - 간결 / 내부 동작 파악 어렵
        - ex) create_stuff_documents_chain, create_retrieval_chain
    
- 고급 Retriever - MultiQueryRetriever, ContextualCompression
    - MultiQueryRetriever
        - 사용자 원본 질문을 LLM으로 여러 변형 질문으로 재작성하고, 검색한 결과를 합쳐서 더 다양한 문서 찾는 기법
        - 각 변형 질문으로 벡터스토어를 검색하고, 모든 검색 결과 합치면서 중복 제거, 최종 문서 리스트 반환
    - ContextualCompressionRetriever
        - 검색된 문서에서 질문과 관련 있는 부분만 추출하여 불필요한 내용 제거 기법
        - 관련 부분만 추출 → LLM에 전달되는 context 더 정확, 토큰 비용 절약 (대신 압축 LLM 호출 비용 O)
        - 일반 검색기는 문서 전체를 반환하지만, 압축검색기는 거기서 관련 부분만 추출하여 LLM에 전달
        - 실무에서는 LLMChainExtractor와 EmbeddingsFilter 중 선택
        - EmbeddingsFilter - 임베딩 유사도 기준으로 문서 필터링하는 압축기, LLM 호출 X라 비용은 적지만 문서 내용 변형 없이 관련 없는 문서 전체를 제거
        - LLMChainExtractor - LLM 사용하여 각 문서에서 질문과 관련된 부분만 추출, LLM 호출 비용이 발생하지만 더 정확함.
    
    <aside>
    💡
    
    정리) 
    
    - MultiQueryRetriever - 질문 다양하게 확장
    - ContextualCompression - 검색 결과에서 핵심만 추출
    - EnsembleRetriever - 벡터 검색과 키워드 검색을 가중 결합
    </aside>
    
- 대화형 RAG - 히스토리 인식 검색
    - 이전 대화 내용을 기억하고, 각 질문의 맥락을 이해해서 정확한 답변을 하도록 하자
    - create_history_aware_retriever
        - 대화 히스토리와 현재 질문을 결합하여 검색 쿼리를 재작성하는 함수
        - 대명사나 문맥 의존 표현을 독립적인 검색 쿼리로 변환
        1. 대화 히스토리 없으면 ⇒ 사용자 질문 그대로 검색기에 전달
        2. 대화 히스토리 있으면 ⇒ LLM이 히스토리 + 현재 질문 분석 ⇒ 독립적인 검색 쿼리 생성
        3. 재작성된 쿼리로 벡터 스토어 검색
    - 히스토리를 유지하면서 문맥 의존적인 질문에도 정확 답변 가능
    - 헬퍼 없이 LCEL로 직접 구현 가능. (RunnablePassthrough.assign(), RunnableBranch)
        - 히스토리 유무에 따라 다르게 처리
        - 이렇게 직접 구현하면 히스토리 처리 로직을 자유롭게 변경할 수 있음. 우선 헬퍼를 쓰고, 특수한 요구사항이 있을 때 직접 구현
- RAG 평가와 품질 개선 전략
    - 평가 두 기준 : 검색 품질 vs 생성 품질
    - 검색 품질
        - Context Relevance
        - Context Recall
    - 생성 품질
        - Faithfulness (Most Important!!)
        - Answer Relevance
    - RAG 품질 개선 전략
        - ex) 엉뚱한 문서 검색 → 검색 품질 낮음 → chunk_size/overlap 조정, 임베딩 모델 변경, MMR 사용
        - ex) 필요한 정보 누락 → 검색 범위 부족 → k값 증가, MultiQueryRetriever 적용
        - ex) 환각 → 생성 충실도 낮음 → 프롬프트 강화, temperature 낮추기
- Reranking
    - 1차 검색(벡터 유사도)으로 얻은 결과를 2차로 다시 정렬하는 기법
        - 1차 검색(벡터 유사도 - 코사인) - 빠르지만 부정확할수도 있음(대략적 관련성)
        - 2차 리랭킹(Cross-Encoder 모델) - 느리지만 정확, 상위 K개만 처리
    - 리랭킹 모델이 질문과 각 문서의 관련성을 더 정밀하게 판단 → 순서를 재정렬
    - 보통 1차 검색으로 20~50개 후보 → 리랭킹으로 상위 3~5개 선별
- chunk_suze 최적화
    - RAG 품질에 가장 큰 영향 미치는 파라미터 - chunk_size, chunk_overlap
    - chunk_size가
        - 작으면 → 정확한 검색(적은 노이즈), 맥락 부족과 문장 단위 분리
        - 크면 → 풍부한 맥락, 노이즈 비중과 토큰 비용 상승