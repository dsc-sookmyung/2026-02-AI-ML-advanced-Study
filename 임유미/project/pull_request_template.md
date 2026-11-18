## 👤 제출자 정보
- **이름**: 임유미
- **팀 / 개인 여부**: (개인 
---

## 📝 프로젝트 개요
### 프로젝트명
<!-- 한 줄로 프로젝트를 표현하는 이름 -->
ShortsCrew: Multi-Agent AI 숏츠 제작 파이프라인

### 한 줄 소개
<!-- 예: 사용자의 일기를 분석해 그날의 감정을 요약해주는 NLP 서비스 -->
도메인만 입력하면 주제 탐색부터 대본·콘티까지 만들어 주는 숏츠 제작 에이전트

### 기획 배경 및 문제 정의
<!-- 어떤 문제를 해결하고 싶은지, 왜 이 주제를 선택했는지 -->
- 숏츠 한 편은 주제 발굴 → 자료 조사 → 대본 → 콘티 → 영상 소스 수집을 매번 반복해야 해서 시간이 많이 듭니다.
- 주제 한 줄을 그대로 LLM에 맡기면 주제가 얕고 뻔해지며, 60초 초과 대본, 출처 없는 수치(환각), 금칙어·형식 오류도 자주 생깁니다.

- 해결 방향
  1. **도메인 → 세부 주제 탐색**: 도메인을 검색하고 깊게 들어가 구체적인 주제를 사용자가 고르게 합니다.
  2. **역할 분리(Multi-Agent)**: 조사, 작성, 검수, 연출, 콘티를 각 에이전트가 맡습니다.
  3. **하이브리드**: 정확해야 하는 검사(길이, 금칙어, 라이선스 필드)는 일반 코드가 담당합니다.
  4. **Human-in-the-loop**: 주제 선택, 대본 검수, 최종 승인에서 사람이 확인합니다.

---

## 🎯 목표
- **핵심 목표**: 도메인 입력만으로 숏츠 제작 패키지(대본·콘티·영상 소스)를 안정적으로 생성하고, 단일 LLM 대비 Multi-Agent 파이프라인이 근거 일치율, 형식 준수율, 품질 점수를 얼마나 높이는지 정량적으로 검증합니다.
- **최소 구현 범위 (MVP)**:
  1. 도메인 입력 → 세부 주제 탐색(사용자 선택, 최대 3단계) → 리서처(사실 3~5개 + 출처) → 작가(대본) 흐름 구현
  2. 검수자(길이·금칙어는 코드, 사실↔출처 대조는 LLM) + 자동 재작업(최대 2회) + 사람 검수(승인 / 수정 요청 / 직접 수정)
  3. 콘티 에이전트 + 비주얼 모드 선택(실제 이미지 소스 검색 / 생성 프롬프트) + 최종 승인 후 JSON·Markdown 저장
  4. 도메인 3개 × 세부 주제 5개(총 15개)로 단일 LLM vs Multi-Agent 비교 평가
  
- **추가로 시도해보고 싶은 것 (Optional)**:
  - CLIP 재정렬 및 기본 순위와의 장면별 Hit@3 비교, 중복 제거, 9:16 세로 크롭 가능 여부 검사
  - 실제 영상 클립 수집 확장, 외부 영상 제작 AI에 콘티를 넘겨 결과 품질 비교
  - Streamlit 간단 UI, TTS·영상·자막 합성, 팀 단위 계층 구조(Hierarchical)
---

## 👉 태스크 & 접근 방법
- **태스크 유형**: <!-- 예: 텍스트 분류 / 감성 분석 / 요약 / 질의응답 / NER / 생성 / RAG 등 --> 
  멀티 에이전트 워크플로 / 도구 호출 / 웹 검색 기반 주제 탐색 / 조건부 텍스트 생성 / 구조화 출력 / 근거 검증 / Human-in-the-loop / (선택) 이미지-텍스트 임베딩 검색 및 재정렬
  

- **사용할 모델 (후보)**: <!-- 예: KoBERT, KoELECTRA, Llama, GPT API 등 -->
  - **생성 LLM**
    - Claude Sonnet 5.5 (`claude-sonnet-5-5`)
    - GPT-5.6 Sol (`gpt-5.6-sol`)
  - 역할별 모델 분리:
    - Supervisor·주제 탐색·검수자의 사실 대조: 경량 모델 Claude Haiku 5.5 (`claude-haiku-5-5`)
    - 작가·콘티: 성능 좋은 모델 Claude Sonnet 5.5
  - 웹 검색: Tavily (`langchain-tavily`) / 이미지 소스 검색: Openverse API (대안: Pexels, Unsplash)
  - **(선택) 이미지-텍스트 매칭**: sentence-transformers의 `clip-ViT-B-32` (이미지 인코더) + `clip-ViT-B-32-multilingual-v1` (한국어 텍스트 인코더)
- **접근 방법**: <!-- 파인튜닝 / 프롬프트 엔지니어링 / RAG / 처음부터 학습 등 -->
  - `create_agent` + 서브에이전트를 도구로 감싸는 subagents 패턴으로 Supervisor 구조를 만듭니다.
  - 흐름: 도메인 입력 → 주제 탐색(최대 3단계) → 리서처 → 작가 → 자동 검수 → 사람 검수 → 콘티 → 소스 수집 또는 프롬프트 생성 → 최종 승인 → 저장
  - 이미지는 "생성"이 아니라 검색 + 재정렬 문제로 접근하고, 코드가 해상도·라이선스·출처 필드를 검사합니다.
  - 각 에이전트는 Pydantic 스키마로 구조화된 결과를 반환하고, 사람 검수·승인은 interrupt 방식으로 구현합니다.
- **선택 이유**:
  - 도메인에서 깊이 들어가는 방식은 뻔한 주제를 피하고 대본의 구체성과 근거 품질을 높입니다.
  - 역할이 뚜렷해 프롬프트 하나보다 에이전트 분리가 유리하고, 정확해야 하는 검사는 코드로 나누는 하이브리드가 안정적입니다.
  - 사람 검수를 자동 검수 뒤, 콘티 앞에 두면 기계적 오류가 걸러진 대본만 읽어 피로가 줄고, 비용이 큰 단계가 수정 때마다 낭비되지 않습니다.
  - 실제 소스와 생성 프롬프트를 모두 제공해 사용자가 저작권 부담과 표현 자유도 중 상황에 맞게 고를 수 있습니다.

---

## 📊 데이터
| 항목 | 내용 |
|---|---|
| 데이터셋 이름 / 출처 | 평가용 도메인·세부 주제 목록, 금칙어 리스트. Tavily 웹 검색 결과(사실·출처 URL), Openverse API 이미지(라이선스·출처 메타데이터 포함). (선택) CLIP 재정렬 평가용 장면별 정답 사진 라벨 |
| 데이터 규모 | 평가 도메인 3개 × 세부 주제 5개 = 15개, 금칙어 30~50개, (선택) 정답 사진 라벨링 장면 30~50개 |
| 언어 | 한국어(대본·자막), 영어(이미지 검색어 병행) |
| 라이선스 / 사용 가능 여부 | 이미지는 상업적 이용과 변경이 허용되는 라이선스만 사용하고 변경 금지(ND)는 제외합니다. 이미지·웹 검색 결과 모두 출처·작성자·라이선스 URL을 함께 저장하며, 각 라이선스와 플랫폼 정책은 사용 전에 직접 확인합니다. |
| 전처리 계획 | 도메인·톤·목표 길이·비주얼 모드를 JSON으로 정리, 금칙어는 텍스트 파일로 관리. 이미지 후보는 라이선스 필터와 해상도·세로 크롭 가능 여부 검사를 거치고, (선택) CLIP 임베딩으로 중복 제거 |

---

## 📏 평가 방법
- **평가 지표**: <!-- 예: Accuracy, F1, ROUGE, BLEU, 사람 평가 등 -->
- **베이스라인**: <!-- 비교 대상이 될 기준 모델 또는 방법 -->

---

## 🛠️ 기술 스택
- **언어 / 프레임워크**: Python, LangChain(`create_agent`), LangGraph(체크포인터, interrupt), Pydantic, Tavily, Openverse API, sentence-transformers(CLIP), Chroma
- **실험 환경**: 로컬 환경, LangSmith, VS Code, Google Colab
- **(선택) 서빙 / 데모**: CLI, Streamlit UI or React

---

## 🗓️ 일정 계획
| 주차 | 계획 |
|---|---|
| 1주차 | 모델 비교·선정, 스키마 설계, 도메인 탐색 에이전트, 작가 에이전트 + 길이·금칙어 검사, 평가용 도메인·금칙어 리스트 작성, 단일 LLM 베이스라인 실행 |
| 2주차 | 리서처·검수자 구현, Supervisor 연결, 자동 재작업 카운터, 콘티 에이전트 구현 (대본·콘티 JSON까지 end-to-end) |
| 3주차 | 대본 사람 검수·최종 승인, 비주얼 모드 2종(실제 소스 검색 / 생성 프롬프트), 오류 처리와 폴백, LangSmith 추적, 베이스라인 ① 평가 |
| 4주차 | 실패 사례 분석, 프롬프트 개선, 블라인드 평가, (여유 시) CLIP 재정렬 실험, README·데모 영상 정리 |
 
- 지연 시 CLIP 재정렬, 스트리밍, StateGraph 비교, 영상 소스 수집 순으로 제외합니다.
---

## ⚠️ 예상 어려움 & 도움이 필요한 부분
<!-- 막힐 것 같은 부분, 리드/팀원에게 피드백 받고 싶은 부분을 자유롭게 적어주세요 -->
## ⚠️ 예상 어려움 & 도움이 필요한 부분

* **Multi-Agent 구조 설계**: 주제 탐색·리서처·작가·검수자·콘티 에이전트의 역할 분리가 적절한지, Supervisor가 순서를 어기거나 불필요하게 호출하는 문제를 어떻게 막고 에이전트 간 결과를 안정적으로 전달할지 고민하고 있습니다. 순서가 중요한 구간은 Supervisor 방식과 순서 고정 방식 중 무엇이 효과적인지도 판단이 필요합니다.
* **근거 검증과 환각 방지**: 리서처가 모은 사실과 출처를 작가가 왜곡 없이 대본에 반영했는지, 검수자(LLM)가 사실↔출처 일치를 일관된 기준으로 판정할 수 있는지 고민하고 있습니다. 검수자의 판정이 틀렸을 때 잘못된 대본이 그대로 통과하는 문제도 걱정됩니다.
* **Human-in-the-loop 흐름 설계**: 주제 선택, 대본 검수, 최종 승인 단계에서 실행을 멈추고 같은 세션으로 재개하는 구조를 어떻게 안정적으로 구현할지 고민하고 있습니다. 자동 재작업과 사람 수정 라운드가 서로 간섭하지 않도록 관리하는 방법과, 사람이 직접 고친 대본이 검수를 우회하는 문제도 고려해야 합니다.
* **이미지 소스 수집**: 한국어 주제를 영어 검색어로 바꿨을 때 검색 품질이 충분한지, 인물·브랜드·최신 뉴스 사진처럼 공개 라이선스 풀에 거의 없는 장면은 어떻게 처리할지 고민하고 있습니다. 라이선스·저작권 해석과 API 호출 제한도 불확실한 부분입니다.
* **평가 설계**: 15개 주제라는 작은 표본으로 단일 LLM과 Multi-Agent의 차이를 의미 있게 비교할 수 있는지, 품질 루브릭(후킹·구성·콘티 적합성)을 어떻게 일관되게 평가할지 고민하고 있습니다. 모델 API가 샘플링 파라미터(temperature 등) 고정을 지원하지 않아 실행 간 편차를 어떻게 통제할지도 피드백을 받고 싶습니다.

---

## 🔗 참고 자료
<!-- 참고한 논문, 블로그, 레포지토리 링크 -->
- LangChain Docs, Migrate from langgraph-supervisor: https://docs.langchain.com/oss/python/migrate/langgraph-supervisor
- LangChain Docs, Subagents: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents
- LangGraph Docs, Interrupts (Human-in-the-loop): https://docs.langchain.com/oss/python/langgraph/interrupts
- LangChain Docs, Structured output: https://docs.langchain.com/oss/python/langchain/structured-output
- LangChain Docs, Tavily search integration: https://docs.langchain.com/oss/python/integrations/tools/tavily_search
- LangGraph Docs, Streaming: https://docs.langchain.com/oss/python/langgraph/streaming
- LangSmith Docs, Trace LangChain applications: https://docs.langchain.com/langsmith/trace-with-langchain
- LangChain Docs, LangGraph v1 migration: https://docs.langchain.com/oss/python/migrate/langgraph-v1
- Openverse API 문서: https://api.openverse.org
- Sentence-Transformers 문서, Image Search (CLIP): https://sbert.net
