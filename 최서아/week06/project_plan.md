## 👤 제출자 정보
- **이름**: 최서아
- **팀 / 개인 여부**: 개인 (추후 팀 구성 가능)

---

## 📝 프로젝트 개요

### 프로젝트명

**LearnTrace — 누적 학습 이력 기반 개인 맞춤형 Multi-Agent 학습 시스템**

### 한 줄 소개

사용자의 질문 이력과 강의안 정보를 연결하고, 여러 Agent가 협업하여 개인 맞춤형 설명과 이해도 검증을 제공하는 AI 학습 서비스

### 기획 배경 및 문제 정의

GPT와 같은 LLM 서비스를 활용하여 공부할 때, 긴 답변을 읽다가 특정 부분에 대해 추가 질문을 하면 새로운 답변이 대화 맨 아래에 생성되어 기존 설명과 추가 답변을 오가야 하는 불편함이 있었다.

또한 꼬리 질문과 답변이 유기적으로 연결·정리되지 않아 동일한 내용을 반복적으로 질문하거나, 과거 학습 내용을 다시 찾는 과정에서 비효율이 발생하였다.

이러한 문제를 해결하기 위해 다음과 같은 서비스를 기획하였다.

- 원본 설명과 꼬리 질문 및 답변을 연결하여 학습 맥락 유지
- 사용자 질문을 개념별로 분류하고 반복적으로 질문하는 부분을 명시적으로 제공
- 답변과 함께 관련 과거 질문 및 강의안의 페이지·근거 제공
- 사용자가 AI에게 직접 개념을 설명하는 TeachBack 학습을 통해 실제 이해도 확인
- 누적 학습 이력을 기반으로 개인별 설명 및 학습 전략 개선

단순히 질문에 답변하는 AI를 넘어, **사용자의 학습 과정을 기억하고 학습 상태에 따라 여러 Agent가 협업하는 개인 맞춤형 학습 시스템**을 구현하고자 한다.

---

## 🎯 목표

### 핵심 목표

- 질문 및 학습 이력의 구조화와 장기 기억 구축
- 강의안과 과거 질문 이력을 활용한 개인화 답변 제공
- 학습 상태에 따른 적응형 Multi-Agent 협업 구현
- Memory 및 Multi-Agent 활용에 따른 성능 비교

### 최소 구현 범위 (MVP)

**1. Learning Analyst Agent — 학습 분석 및 전략 결정**

- 질문의 카테고리, 핵심 개념 및 질문 의도 분류
- 과거 유사 질문 검색 및 반복 질문 탐지
- 학습 이력과 이해도 분석을 통한 학습 전략 추천
- Supervisor 역할로 Teacher 또는 Student Agent를 조건부 호출

**2. Teacher Agent — 개인화 설명 및 평가**

- 강의안과 과거 질문을 활용한 개인 맞춤형 답변 생성
- 관련 강의안 페이지 및 근거 문단 제공
- 현재 질문과 관련된 과거 질문 연결
- TeachBack 과정에서 사용자 설명의 정확성, 누락 개념 및 오개념 평가

**3. Student Agent — TeachBack 학습**

- 사용자가 AI에게 직접 개념을 설명하도록 유도
- 사용자 설명과 과거 학습 이력을 기반으로 꼬리 질문 생성
- Teacher Agent의 평가 결과에 따른 추가 질문 및 이해도 검증

**4. Dual RAG 및 Long-term Memory**

- Document RAG: 강의안 기반 근거 및 관련 페이지 검색
- Memory RAG: 과거 질문, 답변, 오개념 및 이해도 평가 기록 검색
- 학습 결과를 장기 기억에 반영하여 이후 학습에 활용


### 추가로 시도해보고 싶은 것 (Optional)

- 원본 설명의 특정 부분에서 직접 질문하는 인라인 인터페이스
- 질문 트리 및 개념별 지식 그래프 시각화
- 반복 질문 및 이해도 변화 대시보드
- 질문과 답변을 통합한 자동 학습 노트 생성
- 개인별로 효과적인 설명 방식 분석
- ChatGPT 대화 이력 및 기존 학습 노트 연동

---

## 👉 태스크 & 접근 방법

- **태스크 유형:** 텍스트 분류, Semantic Search, Clustering, RAG, 질의응답, LLM-as-a-Judge, Multi-Agent Orchestration

- **사용할 모델 (후보):**
  - LLM: GPT API 또는 Gemini API
  - Embedding: `BAAI/bge-m3`, `paraphrase-multilingual-MiniLM-L12-v2`

### 접근 방법

**1. Supervisor 기반 Multi-Agent**

Learning Analyst Agent가 사용자의 질문과 학습 상태를 분석하여 적절한 Agent와 학습 전략을 결정한다.

- **일반 학습:** 사용자 질문 → Learning Analyst의 질문 분석 → Teacher의 개인화 답변 및 근거 제공 → 학습 기록 저장
- **TeachBack 학습:** 학습 개념 추천 → 사용자 설명 → Student의 꼬리 질문 → Teacher의 이해도 평가 → 추가 질문 또는 보완 설명 → Memory 갱신

Teacher의 평가 결과에 따라 Student의 후속 질문이 달라지고, 학습 결과에 따라 Learning Analyst가 다음 전략을 결정하는 조건부 실행 구조를 구현한다.

**2. Dual RAG**

| 구분 | Document RAG | Memory RAG |
|---|---|---|
| 데이터 | 강의안 PDF, 페이지별 문단 | 과거 질문, 답변, 학습 평가 |
| 목적 | 객관적인 답변 근거 확보 | 사용자 맞춤형 학습 지원 |
| 활용 | 근거 기반 답변 및 평가 | 반복 질문 분석 및 개인화 |

**3. Tool Calling 및 ReAct**

강의안 검색, 과거 질문 검색, 학습 프로필 및 평가 기준 조회 등을 Tool로 정의하고, Agent가 필요한 정보를 판단하여 호출하도록 구성한다.

**4. Guardrails 및 평가**

Structured Output과 개념별 평가 루브릭을 활용하고, 실제 검색된 강의안의 페이지 정보만 출처로 사용하도록 검증한다. 또한 반복 질문 횟수만으로 이해도가 낮다고 판단하지 않도록 설계한다.

### 선택 이유

- **Multi-Agent:** 학습 분석, 설명, 이해도 검증의 역할을 분리하고 평가 결과에 따라 후속 학습 전략을 변경하기 위함
- **RAG / Memory:** 강의안과 과거 학습 이력을 활용하여 근거 기반 개인화 답변을 제공하기 위함
- **LangGraph:** Agent 간 조건부 실행, 상태 관리 및 피드백 루프를 구현하기 위함
- **LLM-as-a-Judge:** 서술형 설명의 정확성 및 오개념을 평가하기 위함

---

## 📊 데이터

| 항목 | 내용 |
|---|---|
| 데이터셋 이름 / 출처 | 자체 구축 학습 질문 데이터셋, 사용자 질문 및 학습 기록, 강의안 PDF |
| 데이터 규모 | 초기 핵심 개념 10~15개, 질문 50~100개, TeachBack 평가 사례 30~50개 구축 목표 |
| 언어 | 한국어 중심, 영어 전공 용어 포함 |
| 라이선스 / 사용 가능 여부 | 직접 작성한 데이터 및 사용 권한이 확보된 학습 자료 활용, 사용자 동의 후 학습 기록 수집 |
| 전처리 계획 | PDF Chunking, 페이지 메타데이터 저장, 개념 및 질문 의도 태깅, 임베딩 생성 |


---

## 📏 평가 방법

### 평가 지표

| 평가 항목 | 평가 지표 |
|---|---|
| 질문 분류 | Accuracy, Macro F1 |
| 유사·반복 질문 탐지 | Precision, Recall, F1 |
| 강의안 근거 검색 | Recall@K, Citation Accuracy |
| 개인화 답변 | 사용자 이력 반영 적합도 |
| 이해도 평가 | 사람 평가와의 일치도, MAE |
| Agent 선택 | Routing Accuracy |
| 시스템 성능 | 응답 시간, 토큰 사용량, LLM 호출 횟수 |

### 베이스라인

| 실험군 | 구성 |
|---|---|
| Baseline A | 단일 LLM + Document RAG |
| Baseline B | 단일 LLM + Document RAG + Memory RAG |
| Baseline C | 순차 실행 Multi-Agent + Dual RAG |
| Proposed | Supervisor 기반 적응형 Multi-Agent + Dual RAG |

동일한 모델과 평가 데이터를 활용하여 다음 항목을 비교하고자 한다.

- **A vs B:** 장기 기억 활용에 따른 개인화 성능 차이
- **B vs C:** 단일 LLM과 Multi-Agent 구조의 성능 차이
- **C vs Proposed:** 고정 실행과 학습 상태에 따른 조건부 실행의 성능 차이

추가로 인라인 인터페이스 구현 시 기존 채팅 방식과 답변 탐색 시간 및 사용자 편의성을 비교해 보고 싶다.

---

## 🛠️ 기술 스택

- **언어 / 프레임워크:** Python, LangChain, LangGraph
- **실험 환경:** 로컬 Python 환경, VS Code, Google Colab
- **서빙 / 데모:** FastAPI, Streamlit 또는 React

---

## 🗓️ 일정 계획

| 주차 | 계획 |
|---|---|
| 1주차 | 요구사항 및 Agent 구조 설계, 초기 데이터셋 구축, 평가 기준 정의 |
| 2주차 | Document RAG, Memory RAG, 질문 분석 및 Teacher Agent 구현 |
| 3주차 | Student Agent 및 TeachBack 구현, Supervisor 기반 조건부 실행 및 피드백 루프 구축 |
| 4주차 | 웹 데모 구현, 베이스라인 비교 평가, 성능 분석 및 결과 정리 |

---

## ⚠️ 예상 어려움 & 도움이 필요한 부분

- **Multi-Agent 구조 설계:** Learning Analyst, Teacher, Student 간의 역할 분리가 적절한지, Supervisor 기반 조건부 실행과 Agent 간 피드백 루프를 어떻게 설계하는 것이 효과적인지 고민하고 있습니다.

- **반복 질문 분석:** 의미적으로 유사한 반복 질문과 심화 학습 목적의 질문을 어떻게 구분하고, 질문 이력과 평가 결과를 활용하여 실제 취약 개념을 판단할 수 있을지 고민하고 있습니다.

- **Long-term Memory 관리:** 전체 대화 이력과 구조화된 학습 기록 중 어떤 저장 방식이 효과적인지, 검색 시 의미적 유사도·최신성·이해도를 어떻게 반영하고 기존 학습 기록을 갱신해야 하는지 고민하고 있습니다.

- **LLM 평가 및 오류 누적 방지:** 사용자 설명과 오개념에 대한 일관된 평가 기준을 마련하고, 잘못된 평가 결과가 장기 기억에 저장되어 이후 Agent의 판단에 영향을 미치는 문제를 방지할 방법을 고민하고 있습니다.

- **개인화 성능 평가:** Memory 및 Multi-Agent 구조가 실제로 개인화 답변과 학습 진단 성능을 개선하는지 정량적으로 검증할 수 있는 평가 지표와 데이터셋 구성 방법에 대해 피드백을 받고 싶습니다.


---

## 🔗 참고 자료

- [DeepTutor: Towards Agentic Personalized Tutoring](https://arxiv.org/abs/2604.26962)
  - 개인화 Multi-Agent 학습 시스템 및 평가 방법 참고
- [DeepTutor GitHub](https://github.com/HKUDS/DeepTutor)
  - RAG, Memory 기반 학습 시스템 아키텍처 및 구현 참고
- [Teach AI How to Code: Using Large Language Models as Teachable Agents for Programming Education](https://doi.org/10.1145/3613904.3642349)
  - Student Agent 및 Learning-by-Teaching 학습 방식 참고


- [LangSmith Evaluation](https://docs.langchain.com/langsmith/evaluation-types)
  - Agent 실행 추적 및 성능 비교 평가 참고
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
  - LLM 기반 평가 방법 및 신뢰성 검증 참고


- [Thinket Demo](https://thinket.ai/app?demo=1&lang=ko)
  - 원본 설명과 꼬리 질문을 연결하는 비선형 학습 인터페이스 참고
- [ChatGPT Study Mode](https://openai.com/index/chatgpt-study-mode/)
  - 꼬리 질문과 단계적 이해도 확인을 통한 능동적 학습 방식 참고