
# Ch.13 멀티 에이전트: Supervisor와 Swarm 정리

> 참고: https://wikidocs.net/333797

## 개념

**멀티 에이전트(Multi-Agent)**는 검색·분석·작성 등 전문 역할을 가진 여러 에이전트가 업무를 나눠 처리하는 구조다.

- 하나의 LLM이 모든 작업을 처리하는 대신, 여러 Agent가 역할을 분담한다.
- Agent마다 서로 다른 모델이 반드시 필요한 것은 아니다.
- 동일 LLM에 서로 다른 시스템 프롬프트와 Tool을 연결해도 된다.
- 각 Agent는 자신의 역할에 맞는 판단과 도구 실행을 수행한다.
- Agent 사이의 작업 전달과 결과 공유가 중요하다.

### 단일 에이전트와 멀티 에이전트

**단일 에이전트**

```text
사용자 요청
    ↓
ReAct Agent
    ├── 검색 Tool
    ├── 분석 Tool
    └── 작성 Tool
    ↓
최종 응답
```

- 하나의 Agent가 모든 도구를 관리한다.
- 구조가 단순하고 구현이 쉽다.
- 업무가 많아지면 프롬프트와 도구 관리가 복잡해질 수 있다.

**멀티 에이전트**

```text
사용자 요청
    ↓
Supervisor
    ├── Research Agent
    ├── Analyst Agent
    └── Writer Agent
    ↓
최종 응답
```

- Agent마다 전문 역할을 담당한다.
- 도구와 프롬프트를 역할별로 분리할 수 있다.
- 독립적인 개발과 유지보수가 가능하다.
- 대신 Agent 사이의 통신과 상태 관리가 필요하다.

### 멀티 에이전트를 사용하는 이유

1. **전문 역할 분리**
   - Researcher는 정보 검색만 담당
   - Analyst는 데이터 분석 담당
   - Writer는 보고서 작성 담당

2. **도구 관리**
   - 각 Agent에게 필요한 Tool만 제공할 수 있다.
   - Tool이 많을 때 선택 오류를 줄이는 데 도움이 된다.

3. **협업과 확장성**
   - 독립적인 작업을 병렬로 실행하도록 설계할 수 있다.
   - 새로운 전문 Agent를 추가하기 편리하다.

4. **유지보수**
   - 역할별 프롬프트를 독립적으로 수정할 수 있다.
   - 서로 다른 개발팀이 Agent를 담당할 수 있다.

> 멀티 에이전트는 필수 기술이 아니라, 복잡한 업무를 역할별로 관리하기 위한 아키텍처다.


## 주요 패턴

| 패턴 | 업무 전달 | 사례 |
|---|---|---|
| **Supervisor** | 중앙 관리자 → 전문 Agent → 관리자 | 조사 후 보고서 작성 |
| **Swarm** | 전문 Agent끼리 직접 Handoff | 고객 상담 부서 전환 |
| **Hierarchical** | 상위 Supervisor → 하위 팀 Supervisor → Agent | 대규모 업무 팀 |

### ① Supervisor

- 중앙 Supervisor가 전체 작업을 관리한다.
- 사용자 요청에 맞는 전문 Agent를 선택한다.
- Agent의 결과를 받아 다음 작업을 결정한다.
- 모든 작업이 완료되면 최종 응답을 생성한다.

```text
사용자
  ↓
Supervisor
  ↓
Researcher
  ↓
Supervisor
  ↓
Writer
  ↓
Supervisor
  ↓
최종 응답
```

**특징:** 전문 Agent는 작업 완료 후 Supervisor에게 돌아온다.

### ② Swarm

- 중앙 관리자 없이 Agent끼리 직접 업무를 전달한다.
- 현재 Agent가 다음 담당 Agent를 선택한다.
- 상담·업무 인계처럼 담당자가 변경되는 서비스에 적합하다.

```text
사용자
  ↓
General Agent
  ↓
Billing Agent
  ↓
Technical Agent
  ↓
최종 응답
```

**특징:** Agent 사이에서 직접 Handoff가 발생한다.

### ③ Hierarchical

- Supervisor를 여러 계층으로 구성한다.
- 상위 Supervisor가 팀 단위의 하위 Supervisor를 관리한다.
- 하위 Supervisor는 전문 Agent를 관리한다.

```text
Main Supervisor
    ├── Research Supervisor
    │       ├── Search Agent
    │       └── Analyst Agent
    │
    └── Writing Supervisor
            ├── Writer Agent
            └── Reviewer Agent
```

**특징:** Agent의 수가 많은 복잡한 시스템에 적합하다.



## Supervisor

**Supervisor**는 어떤 Agent에게 작업을 맡길지 판단하고 결과를 취합하는 중앙 관리자다.


### 주요 함수

**① `create_agent()`**

- 독립적인 Agent를 생성한다.
- LLM, Tool, 시스템 프롬프트를 지정한다.
- 각 Agent는 자신의 역할에 맞게 작업한다.

| 인자 | 역할 |
|---|---|
| `model` | 사용할 LLM |
| `tools` | 사용할 도구 |
| `name` | Agent 이름 |
| `system_prompt` | Agent 역할과 지침 |

**② `create_supervisor()`**

- 여러 Agent를 관리하는 상위 그래프를 만든다.
- `agents`에 관리할 Agent들을 등록한다.
- `model`은 Supervisor의 의사결정에 사용된다.

**③ `compile()`**

- 그래프를 실행 가능한 형태로 만든다.
- 이후 `invoke()`로 실행할 수 있다.

### 실행 예시

```python
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "AI 시장을 조사하고 보고서를 작성해줘"
        }
    ]
})
```

실행 흐름은 다음과 같다.

1. Supervisor가 사용자 요청을 분석한다.
2. Researcher에게 조사를 맡긴다.
3. Researcher가 검색 Tool을 실행한다.
4. 조사 결과가 Supervisor에게 전달된다.
5. Supervisor가 Writer에게 작업을 맡긴다.
6. Writer가 보고서를 작성한다.
7. Supervisor가 최종 결과를 반환한다.

### Handoff와 Tool Calling

Supervisor는 다른 Agent에게 작업을 넘길 때 Handoff Tool을 활용할 수 있다.

```text
Supervisor
    ↓
transfer_to_researcher
    ↓
Researcher 실행
    ↓
Supervisor로 복귀
```

- `transfer_to_researcher`: Researcher에게 제어권 전달
- `transfer_to_writer`: Writer에게 제어권 전달

일반 Tool Calling과의 차이는 다음과 같다.

| 구분 | Tool Calling | Handoff |
|---|---|---|
| 목적 | 특정 기능 실행 | 다른 Agent로 전환 |
| 대상 | 함수·API | Agent |
| 예시 | `multiply(5,4)` | `transfer_to_writer` |

**Handoff는 Tool Calling을 이용해 그래프의 실행 제어권을 변경하는 방식이다.**

### 커스텀 프롬프트

```python
custom_prompt = """
당신은 Supervisor입니다.

1. researcher에게 자료 조사를 맡기세요.
2. writer에게 보고서 작성을 맡기세요.
3. 결과를 검토한 뒤 최종 답변하세요.
"""

workflow = create_supervisor(
    agents=[researcher, writer],
    model=model,
    prompt=custom_prompt
)
```

- 프롬프트로 작업 순서를 안내할 수 있다.
- 그러나 LLM이 반드시 지시한 순서대로 실행한다고 보장할 수는 없다.
- 강제적인 순서는 LangGraph의 엣지나 조건 분기로 구현한다.


## Swarm

**Swarm**은 중앙 Supervisor 없이 Agent들이 서로 직접 업무를 전달하는 구조다.

### Handoff

**Handoff**는 현재 Agent가 다른 Agent에게 실행 제어권을 넘기는 것이다.

```text
사용자: 환불이 가능한가요?
    ↓
General Agent
    ↓
환불 문의로 판단
    ↓
Billing Agent에게 Handoff
    ↓
Billing Agent가 답변
```

### Handoff 실행 과정

1. 현재 Agent가 질문을 분석한다.
2. 다른 Agent가 필요하다고 판단한다.
3. Handoff Tool을 호출한다.
4. LangGraph가 다음 실행 Agent를 변경한다.
5. 새로운 Agent가 작업을 수행한다.


### 핵심 함수

**① `create_handoff_tool()`**

```python
to_billing = create_handoff_tool(
    agent_name="billing_agent",
    description="결제 및 환불 문의를 담당합니다."
)
```

- 다른 Agent로 전환하는 도구를 생성한다.
- `agent_name`: 전환 대상
- `description`: 도구 사용 조건을 설명

**② `create_swarm()`**

- 여러 Agent를 하나의 협업 구조로 만든다.
- `agents`: 참여할 Agent 목록
- `default_active_agent`: 최초 실행 Agent

**③ `active_agent`**

- 현재 작업을 수행하는 Agent를 나타낸다.
- Handoff 이후 변경된다.
- 상태를 저장하면 다음 질문에서도 이어서 응답할 수 있다.

### Checkpointer와 thread_id

**Checkpointer**는 그래프의 상태를 저장한다.

```python
app = workflow.compile(
    checkpointer=InMemorySaver()
)
```

**`thread_id`**는 대화 세션을 식별한다.

```python
config = {
    "configurable": {
        "thread_id": "customer_001"
    }
}
```

```text
첫 번째 질문
    ↓
General Agent
    ↓
Billing Agent로 Handoff
    ↓
상태 저장
    ↓
두 번째 질문
    ↓
Billing Agent가 이어서 응답
```

- 같은 `thread_id`를 사용하면 기존 상태를 이어서 활용할 수 있다.
- `InMemorySaver`는 메모리에 저장한다.
- 프로그램 재시작 시 데이터가 사라질 수 있다.
- 실제 운영에서는 영속적인 저장소를 사용하는 것이 적합하다.


## 상태와 메시지

멀티 에이전트에서는 Agent 사이의 정보 공유가 중요하다.

### Message Passing

**Message Passing**은 Agent 사이에 메시지와 작업 결과를 전달하는 것이다.

```text
Researcher
    ↓
조사 결과 생성
    ↓
Supervisor
    ↓
Writer에게 전달
    ↓
보고서 작성
```

### MessagesState

- `MessagesState`: 메시지 목록을 공통 상태로 관리
- 사용자 메시지, Agent 응답, Tool 결과 등을 저장
- 다른 Agent가 이전 작업 맥락을 참고할 수 있도록 지원


### output_mode

```python
workflow = create_supervisor(
    agents=[researcher, writer],
    model=model,
    output_mode="full_history"
)
```

| 설정 | 내용 | 특징 |
|---|---|---|
| `full_history` | 하위 Agent의 실행 이력 전달 | 상세 추적 가능 |
| `last_message` | 마지막 메시지 중심 전달 | 토큰 절약 가능 |

- `full_history`는 중간 실행 과정을 확인하는 데 유용하다.
- `last_message`는 필요한 결과만 전달하는 데 적합하다.
- 이 설정은 하위 Agent가 상위 상태에 전달하는 메시지 범위와 관련된다.


### ReAct와 Multi-Agent 비교

| 항목 | ReAct | Multi-Agent |
|---|---|---|
| 중심 | 하나의 Agent | 여러 Agent |
| 선택 대상 | Tool | Agent |
| 주요 동작 | 도구 실행·재판단 | 업무 위임·협력 |
| 주요 기술 | ToolNode | Handoff |
| 목적 | 반복적인 도구 사용 | 전문 역할 분담 |

**핵심:** ReAct는 '한 Agent가 어떤 Tool을 사용할까?'이고, 멀티 Agent는 '어떤 Agent에게 업무를 맡길까?'이다.

멀티 에이전트는 ReAct를 대체하는 개념이 아니라, 여러 전문 Agent가 협력하도록 확장하는 구조다.
