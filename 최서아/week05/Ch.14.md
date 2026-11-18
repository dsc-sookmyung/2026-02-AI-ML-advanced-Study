
# Ch.14 LangSmith: 디버깅·평가·모니터링 정리

> 참고: https://wikidocs.net/333798

## LangSmith란?

**LangSmith**는 LLM·RAG·ReAct·멀티 에이전트가 **어떻게 실행되었는지 기록하고, 답변 품질을 평가하며, 운영 상태를 모니터링**하는 플랫폼이다.

- LangChain 팀에서 개발한 LLM 애플리케이션 관측·평가 도구다.
- LangChain 외의 Python 프로그램에도 적용할 수 있다.
- AI의 실행 과정을 확인하고 문제의 원인을 분석할 수 있다.
- 프롬프트나 모델 변경 전후의 성능을 비교할 수 있다.

### LangSmith가 필요한 이유

AI 서비스에서는 답변이 잘못되었을 때 원인을 파악하기 어려울 수 있다.

예를 들어 다음과 같은 시스템을 생각해 보자.

```text
사용자 요청
    ↓
Supervisor
    ↓
Research Agent
    ↓
Search Tool
    ↓
Writer Agent
    ↓
최종 보고서
```

보고서에 잘못된 정보가 포함되었다면 다음을 확인해야 한다.

- Search Tool이 잘못된 자료를 검색했는가?
- Research Agent가 자료를 잘못 해석했는가?
- Supervisor가 잘못된 Agent를 선택했는가?
- Writer Agent가 근거 없는 내용을 생성했는가?

**LangSmith는 각 실행 단계의 기록을 확인하여 오류 원인을 추적할 수 있게 한다.**

## 핵심 기능 4가지

| 기능 | 질문 | 주요 내용 |
|---|---|---|
| **Tracing** | 어떻게 답했나? | LLM·Tool 호출, 입력/출력, 실행시간, 오류 기록 |
| **Evaluation** | 답이 좋은가? | 테스트 데이터셋과 평가자로 성능 측정 |
| **Prompt Hub** | 어떤 프롬프트인가? | 프롬프트 저장·공유·버전관리 |
| **Monitoring** | 운영 중 잘 작동하나? | 지연·오류·토큰·비용·사용자 피드백 확인 |

### ① Tracing

- LLM과 Tool의 실행 과정을 기록한다.
- 입력, 출력, 소요 시간, 오류를 확인할 수 있다.
- AI가 예상과 다르게 동작한 원인을 분석한다.

### ② Evaluation

- AI 답변의 정확도와 품질을 측정한다.
- 평가 데이터셋과 평가 기준을 활용한다.
- 모델이나 프롬프트의 버전을 비교한다.

### ③ Prompt Hub

- 프롬프트를 저장하고 공유한다.
- 변경 이력과 버전을 관리한다.
- 검증된 프롬프트를 재사용할 수 있다.

### ④ Monitoring

- 배포된 AI 서비스의 상태를 확인한다.
- 응답시간, 오류율, 토큰 사용량 등을 관찰한다.
- 사용자 피드백을 수집할 수 있다.


## Tracing

**Tracing**은 AI가 어떤 과정을 거쳐 답변했는지 추적하는 기능이다.

### 주요 개념

- **Trace**: 사용자 요청 한 건의 전체 실행 기록
- **Run**: 개별 모델·도구·함수 실행
- **Run Tree**: Run의 부모·자식 관계
- **Span**: 각 실행이 차지한 시간 구간

### 실행 예시

```text
Trace: 사용자 질문
    │
    ├── Run 1: LLM 판단
    │
    ├── Run 2: Weather Tool 실행
    │
    ├── Run 3: LLM 재판단
    │
    ├── Run 4: Calculator Tool 실행
    │
    └── Run 5: 최종 답변
```

각 Run에서 확인할 수 있는 정보는 다음과 같다.

- 입력 데이터
- 출력 데이터
- 실행 시간
- 오류 여부
- 지원되는 경우 토큰 사용량과 비용


### LangChain 실행 추적

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="gpt-4.1-mini"
)

response = model.invoke(
    "RAG가 무엇인지 설명해 줘."
)

print(response.content)
```

- LLM 실행 기록을 LangSmith로 전송하도록 설정할 수 있다.
- 웹 대시보드에서 해당 호출의 입력과 출력을 확인한다.
- 실행 시간과 토큰 사용량 등을 분석할 수 있다.

### @traceable

`@traceable`은 직접 작성한 Python 함수의 실행을 추적하기 위한 데코레이터다.

```python
from langsmith import traceable

@traceable(name="문서 검색")
def search_document(query: str):
    return f"검색 결과: {query}"
```

### 여러 함수의 실행 추적

```python
from langsmith import traceable

@traceable(name="문서 검색")
def search_document(query: str):
    return f"{query} 관련 문서"

@traceable(name="RAG 파이프라인")
def rag_pipeline(question: str):
    context = search_document(question)

    return {
        "answer": f"검색 결과: {context}"
    }

result = rag_pipeline("장학금 신청 조건")
```

실행 기록은 다음과 같이 표현될 수 있다.

```text
RAG 파이프라인
    ↓
문서 검색
    ↓
검색 결과 반환
    ↓
최종 결과
```

### @tool과 @traceable 차이

| 구분 | @tool | @traceable |
|---|---|---|
| 목적 | 도구 정의 | 실행 기록 |
| 사용 대상 | LLM Tool Calling | Python 함수 |
| 주요 역할 | 외부 기능 호출 | 실행 과정 관찰 |

**`@tool`은 기능을 정의하고, `@traceable`은 그 실행을 관찰한다.**

### Tracing 활용

RAG에서 잘못된 답변이 나왔다면 다음을 확인한다.

1. 사용자 질문 확인
2. Retriever 검색 결과 확인
3. LLM에 전달된 Context 확인
4. LLM이 생성한 답변 확인
5. 오류 원인 분석

검색 단계에서 문제가 발생했다면 Retriever를 개선하고, 생성 단계에서 문제가 발생했다면 프롬프트나 모델을 개선할 수 있다.

---

## Evaluation

**Evaluation**은 AI의 답변이 얼마나 정확하고 적절한지 평가하는 과정이다.

### 평가 과정

1. **Dataset**: 질문과 기대 정답 저장.
2. **Target**: 평가할 Agent 또는 체인 실행.
3. **Evaluator**: 정답 비교·의미 평가·사람 평가 등의 기준 적용.
4. **Experiment**: 프롬프트·모델 버전별 결과 비교.

```text
Dataset
   ↓
Target 실행
   ↓
AI 답변 생성
   ↓
Evaluator
   ↓
평가 점수
   ↓
Experiment 결과 저장
```

### ① Dataset

평가 데이터셋은 질문과 기대 정답을 모아둔 자료다.

| 질문 | 기대 정답 |
|---|---|
| RAG의 약자는? | Retrieval-Augmented Generation |
| LangGraph의 핵심 구조는? | StateGraph |
| Tool Calling이란? | 외부 도구 호출 요청 |


### ② Target

Target은 평가할 AI 프로그램이다.

```python
def my_agent(inputs: dict) -> dict:
    question = inputs["question"]

    if "RAG" in question:
        return {
            "answer": "Retrieval-Augmented Generation"
        }

    return {
        "answer": "StateGraph"
    }
```

실제 프로젝트에서는 RAG 체인이나 ReAct Agent를 평가 대상으로 사용할 수 있다.

### ③ Evaluator

Evaluator는 AI 답변을 평가하는 기준이다.

| 방법 | 특징 | 한계 |
|---|---|---|
| Exact Match | 문자열 일치 비교 | 표현 차이에 취약 |
| LLM-as-Judge | LLM이 의미 평가 | 판단 오류·비용 |
| Human Evaluation | 사람이 직접 평가 | 시간 소요 |
| Custom Evaluator | 사용자 정의 함수 | 별도 구현 필요 |

### Exact Match

**Exact Match**는 생성된 답변과 정답이 정확히 같은지 비교한다.

```python
def exact_match(outputs, reference_outputs):
    predicted = outputs["answer"].strip().lower()
    expected = reference_outputs["answer"].strip().lower()

    score = 1.0 if predicted == expected else 0.0

    return {
        "key": "exact_match",
        "score": score
    }
```

| 출력 | 정답 | 점수 |
|---|---|---|
| StateGraph | StateGraph | 1.0 |
| State Graph | StateGraph | 0.0 |
| 핵심 구조는 StateGraph입니다. | StateGraph | 0.0 |

**한계:** 의미는 같아도 표현이 다르면 오답 처리될 수 있다.

### LLM-as-Judge

**LLM-as-Judge**는 다른 LLM을 평가자로 활용하는 방식이다.

```text
기대 정답:
LangGraph의 핵심 구조는 StateGraph입니다.

AI 출력:
LangGraph는 StateGraph를 이용해
상태 기반 실행 흐름을 구성합니다.
```

- 문자열은 달라도 의미가 일치하는지 판단할 수 있다.
- 복잡한 자연어 응답 평가에 유용하다.
- 다만 평가 모델도 잘못 판단할 수 있다.
- 추가 LLM 호출 비용이 발생할 수 있다.

### Human Evaluation

- 사람이 직접 AI의 답변을 평가한다.
- 전문 지식이 필요한 결과에 적합하다.
- 신뢰성이 중요한 서비스에서 유용하다.
- 대량 평가에는 시간과 비용이 많이 든다.

### ④ evaluate()로 자동 평가

```python
from langsmith import evaluate

results = evaluate(
    my_agent,
    data="agent-evaluation",
    evaluators=[exact_match],
    experiment_prefix="agent-v1"
)
```

| 인자 | 역할 |
|---|---|
| `target` | 평가할 함수·체인 |
| `data` | 평가 데이터셋 |
| `evaluators` | 평가 기준 |
| `experiment_prefix` | 실험 이름 |
| `max_concurrency` | 동시 실행 수 |

### Experiment

프롬프트나 모델을 변경했다면 동일한 데이터셋으로 평가한다.

| 지표 | v1 | v2 |
|---|---|---|
| 정답 정확도 | 75% | 88% |
| 출처 정확도 | 65% | 92% |
| 응답시간 | 2.1초 | 2.5초 |

※ 수치는 설명을 위한 가상 예시다.

- v2는 정확도가 개선되었다.
- 하지만 응답시간은 증가했다.
- 따라서 정확도, 비용, 속도를 함께 비교해야 한다.

### RAG 평가 지표

- **Retrieval Relevance**: 검색 문서가 질문과 관련 있는가?
- **Faithfulness**: 답변이 검색된 문서에 근거하는가?
- **Answer Correctness**: 정답과 일치하는가?
- **Citation Accuracy**: 출처가 정확한가?

RAG에서는 단순 정답률뿐 아니라 검색 품질과 근거 충실성도 중요하다.


## Prompt Hub

**Prompt Hub**는 프롬프트를 저장하고 공유하며 버전을 관리하는 기능이다.

### 필요한 이유

- 프롬프트 변경 이력을 관리할 수 있다.
- 코드와 프롬프트를 분리할 수 있다.
- 여러 개발자가 동일한 프롬프트를 사용할 수 있다.
- 프롬프트 버전별 평가 결과를 비교할 수 있다.

### 프롬프트 저장

```python
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

client = Client()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 장학금 상담 AI입니다. "
        "제공된 문서만 참고하여 답변하세요."
    ),
    ("human", "{question}")
])

client.push_prompt(
    "scholarship-assistant",
    object=prompt
)
```

- `push_prompt()`: 프롬프트 등록
- `ChatPromptTemplate`: 메시지 템플릿 구성

### 프롬프트 불러오기

```python
prompt = client.pull_prompt(
    "scholarship-assistant"
)
```

- `pull_prompt()`: 저장된 프롬프트 불러오기
- 동일한 프롬프트를 여러 프로그램에서 재사용할 수 있다.

### 버전 관리 예시

```text
Prompt v1
    ↓
검색 문서 참고

Prompt v2
    ↓
검색 문서 참고 + 출처 표시

Prompt v3
    ↓
출처 표시 + 근거 부족 시 추측 금지
```

프롬프트 버전과 평가 결과를 연결하면 변경 효과를 추적할 수 있다.

실제 운영 환경에서는 검증된 특정 버전을 고정하여 사용하는 것이 안전하다.


## Monitoring

**Monitoring**은 배포된 AI 서비스가 정상적으로 작동하는지 지속적으로 확인하는 기능이다.

### 주요 지표

- 요청 수(Request Volume)
- 응답시간(Latency)
- **P95 Latency**
- 오류율(Error Rate)
- 토큰 사용량(Token Usage)
- API 비용
- 사용자 피드백 점수

| 지표 | 의미 | 활용 |
|---|---|---|
| Request Volume | 요청 수 | 이용량 분석 |
| Latency | 응답시간 | 성능 확인 |
| P95 Latency | 95번째 백분위 응답시간 | 느린 요청 파악 |
| Error Rate | 오류 비율 | 안정성 확인 |
| Token Usage | 토큰 사용량 | 비용 관리 |
| Feedback Score | 사용자 평가 | 만족도 분석 |

### P95 Latency

P95 Latency는 전체 요청 응답시간의 95번째 백분위수다.

예를 들어 100건 중 약 95건이 4초 이내에 완료된다면 P95 Latency는 약 4초다.

- 평균 응답시간만으로는 느린 요청을 파악하기 어렵다.
- P95를 함께 보면 일부 사용자의 지연 문제를 확인할 수 있다.

### 사용자 피드백 저장

```python
from langsmith import Client

client = Client()

client.create_feedback(
    run_id="실제_실행_UUID",
    key="thumbs_up",
    score=0,
    comment="잘못된 정보가 포함되었습니다."
)
```

- `run_id`: 실행 기록 식별자
- `key`: 피드백 항목
- `score`: 평가 점수
- `comment`: 사용자 의견

### 피드백 활용 흐름

```text
사용자 질문
    ↓
AI 답변
    ↓
사용자 피드백
    ↓
해당 Trace 확인
    ↓
문제 원인 분석
    ↓
프롬프트·검색 로직 개선
    ↓
Evaluation 재실행
```

**사용자 피드백을 저장했다고 모델이 자동으로 재학습하는 것은 아니다.**

피드백은 개선을 위한 데이터이며 실제 수정과 검증은 별도로 진행해야 한다.

### 주의사항

- 개인정보가 로그에 기록되지 않도록 관리한다.
- 인증 토큰과 API 키를 저장하지 않는다.
- 필요하면 입력·출력 데이터를 마스킹한다.
- 오류율, 응답시간, 비용을 함께 확인한다.

---

## 전체 연결

지금까지 학습한 기술의 관계는 다음과 같다.

```text
Tool Calling
    ↓
외부 도구 호출
    ↓
RAG
    ↓
외부 문서 검색
    ↓
ReAct
    ↓
도구 사용·재판단 반복
    ↓
Multi-Agent
    ↓
여러 Agent의 협력
    ↓
LangSmith
    ↓
실행 추적·평가·운영 관리
```

### LangSmith를 이용한 품질 개선 과정

```text
AI 서비스 구현
    ↓
Tracing으로 실행 확인
    ↓
오류 원인 분석
    ↓
프롬프트·로직 수정
    ↓
Evaluation으로 성능 검증
    ↓
개선 버전 배포
    ↓
Monitoring으로 운영 상태 확인
```

### 실제 RAG 챗봇 활용 예시

장학금 상담 AI가 잘못된 학점 기준을 알려주었다고 가정하자.

**① Tracing**

- 어떤 문서를 검색했는지 확인한다.
- LLM에게 어떤 Context가 전달되었는지 확인한다.

**② Prompt Hub**

- 최신 규정을 우선 참고하도록 프롬프트를 개선한다.
- 수정한 프롬프트를 새 버전으로 저장한다.

**③ Evaluation**

- 기존 질문 데이터셋으로 재평가한다.
- 정확도와 출처 일치 여부를 비교한다.

**④ Monitoring**

- 배포 후 유사한 오류가 재발하는지 확인한다.
- 사용자 피드백과 응답시간을 모니터링한다.

### 기능 비교

| 질문 | 필요한 기능 |
|---|---|
| 어떤 Tool을 호출했는가? | Tracing |
| 어디서 오류가 발생했는가? | Tracing |
| 답변이 정확한가? | Evaluation |
| 모델 변경 후 성능이 좋아졌는가? | Evaluation |
| 어떤 프롬프트를 사용했는가? | Prompt Hub |
| 서비스 응답시간이 느려졌는가? | Monitoring |

### 핵심 함수

| 함수·설정 | 역할 |
|---|---|
| `LANGSMITH_TRACING=true` | Tracing 활성화 |
| `@traceable` | 함수 실행 추적 |
| `Client()` | LangSmith 클라이언트 |
| `create_dataset()` | 평가 데이터셋 생성 |
| `create_examples()` | 평가 데이터 등록 |
| `evaluate()` | 자동 평가 |
| `push_prompt()` | 프롬프트 저장 |
| `pull_prompt()` | 프롬프트 불러오기 |
| `create_feedback()` | 사용자 피드백 기록 |

**핵심:** LangSmith는 Agent를 만드는 도구라기보다, **만든 시스템의 실행과 품질을 관측·개선하는 도구**다.

즉, 이전 챕터가 AI 시스템의 구현 방법이었다면 Ch.14는 실제 시스템의 품질을 검증하고 안정적으로 운영하는 방법을 배우는 과정이다.
