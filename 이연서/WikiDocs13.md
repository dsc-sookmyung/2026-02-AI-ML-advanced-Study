# 위키독스 13

개념: 멀티 에이전트 시스템: Supervisor와 Swarm

## 1. 멀티 에이전트 아키텍처

### 1. Multi-Agent System이란?

**Multi-Agent System**은 여러 개의 전문화된 AI Agent가 서로 협력하여 복잡한 작업을 수행하는 시스템이다.

각 Agent는 고유한 역할, 도구, 프롬프트를 가질 수 있으며, 작업을 분담하여 수행한다.

**Single-Agent와 Multi-Agent의 차이**

- `Single-Agent`**:** 하나의 Agent가 여러 도구를 사용하며 작업 수행
- `Multi-Agent`**:** 여러 Agent가 역할을 나누고 서로 협력하여 작업 수행

Multi-Agent의 핵심은 단순히 Agent의 개수가 많다는 것이 아니라, 역할과 의사결정을 분리하여 협업하도록 구성한다는 점이다.

### 2. Multi-Agent의 주요 패턴

#### ① Supervisor — 중앙 지휘형

중앙 Supervisor가 사용자 요청을 분석하고 적절한 전문 Agent에게 작업을 위임하는 구조이다.

- Supervisor가 다음에 실행할 Agent를 결정한다.
- 전문 Agent는 담당 작업을 수행한다.
- 작업 결과가 Supervisor에게 반환된다.
- Supervisor가 다음 작업을 위임하거나 최종 답변을 생성한다.

**핵심: 중앙 Supervisor가 작업 흐름을 조율한다.**

#### ② Swarm — 자율 협업형

중앙 Supervisor 없이 각 Agent가 직접 다른 Agent에게 작업을 전달하는 구조이다.

- 현재 실행 중인 Agent가 다음 Agent를 선택한다.
- Agent 간 직접 Handoff가 가능하다.
- Handoff는 데이터뿐 아니라 실행 제어권도 전달한다.

**핵심: Agent가 직접 다음 Agent로 제어권을 넘긴다.**

#### ③ Hierarchical — 계층형

여러 Supervisor를 계층적으로 구성하여 복잡한 작업을 관리하는 구조이다.

- 상위 Supervisor가 여러 팀을 관리한다.
- 각 팀의 Supervisor가 해당 팀의 전문 Agent를 관리한다.
- 규모가 크고 조직적인 협업이 필요한 시스템에 활용할 수 있다.

**핵심: Supervisor를 다른 Supervisor가 관리하는 계층적 구조이다.**

### 3. 패턴 비교

| 구분 | Supervisor | Swarm | Hierarchical |
| --- | --- | --- | --- |
| 제어 방식 | 중앙 집중 | 분산 | 계층적 |
| 다음 Agent 결정 | Supervisor | 현재 Agent | 계층별 Supervisor |
| 핵심 특징 | 중앙 조율 | 직접 Handoff | 팀 단위 조율 |
| 적합한 상황 | 작업 순서·통제 중요 | 자율적인 Agent 전환 | 대규모 시스템 |

### 4. Multi-Agent의 장단점

**장점**

- 역할별 전문성 확보
- Agent별 프롬프트 및 도구 관리
- 복잡한 작업의 분담과 확장

**단점**

- 여러 LLM 호출로 비용 증가 가능
- Agent 간 통신 및 상태 관리 복잡
- 작업 위임과 결과 전달 과정에서 오류 발생 가능

### 핵심 정리

Multi-Agent System은 여러 전문 Agent의 협업을 통해 복잡한 문제를 해결하는 구조이다.

- **Supervisor:** 중앙에서 결정
- **Swarm:** 각 Agent가 직접 결정
- **Hierarchical:** 여러 계층의 Supervisor가 결정

단순한 작업은 Single-Agent로도 해결할 수 있으므로 Multi-Agent를 무조건 도입하기보다 작업의 복잡도와 역할 분리 필요성을 고려해야 한다.

## 2. Supervisor와 `create_supervisor()`

### 1. Supervisor 패턴이란?

**Supervisor 패턴**은 중앙 Supervisor Agent가 사용자 요청을 분석하고, 적절한 전문 Agent에게 작업을 위임한 뒤 결과를 취합하는 멀티 에이전트 구조이다.

**실행 흐름**

![스크린샷 2026-11-11 오전 3.11.23.png](%EC%9C%84%ED%82%A4%EB%8F%85%EC%8A%A4%2013/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-11-11_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_3.11.23.png)

1. Supervisor가 사용자 요청 분석
2. 적절한 전문 Agent 선택
3. Handoff Tool을 호출하여 제어권 전달
4. 전문 Agent가 필요한 작업 수행
5. 작업 결과를 Supervisor에게 반환
6. Supervisor가 추가 작업을 위임하거나 최종 답변 생성

### 2. Handoff Tool과 일반 Tool Calling

| 구분 | Handoff Tool | 일반 Tool |
| --- | --- | --- |
| 목적 | Agent 간 작업 위임 | 특정 기능 실행 |
| 실행 관계 | Agent → Agent | Agent → Tool |
| 예시 | `transfer_to_researcher` | `search_web` |

**Handoff Tool**은 다른 Agent에게 제어권을 전달하는 도구이다.

**일반 Tool**은 해당 Agent가 필요한 기능을 실행하기 위해 사용하는 도구이다.

Supervisor 역시 Tool Calling 메커니즘을 활용하여 전문 Agent를 선택할 수 있다.

### 3. 전문 Agent 생성

```python
researcher = create_agent(
    model=model,
    tools=[search_web],
    name="researcher",
    system_prompt="당신은 자료 조사 전문가입니다."
)

writer = create_agent(
    model=model,
    tools=[write_document],
    name="writer",
    system_prompt="당신은 보고서 작성 전문가입니다."
)
```

- `model`: Agent가 사용할 LLM
- `tools`: Agent가 사용할 수 있는 도구
- `name`: Agent를 구분하기 위한 고유 이름
- `system_prompt`: Agent의 역할 및 행동 지침

같은 LLM을 사용하더라도 도구와 프롬프트를 다르게 설정하면 서로 다른 역할의 Agent를 구성할 수 있다.

### 4. `create_supervisor()` 구현

```python
workflow = create_supervisor(
    agents=[researcher, writer],
    model=model,
    prompt=(
        "당신은 Supervisor입니다. "
        "자료 조사는 researcher에게, "
        "보고서 작성은 writer에게 위임하세요."
    )
)

app = workflow.compile()
```

**주요 파라미터**

- `agents` : 관리할 전문 Agent 목록
- `model` : Supervisor가 사용하는 LLM
- `prompt` : 업무 위임 기준과 행동 지침
- `output_mode` : 전문 Agent의 메시지 기록 반영 방식
- `supervisor_name` : Supervisor 이름

`create_supervisor()`는 등록된 전문 Agent에 대응하는 Handoff Tool을 생성하여 Supervisor가 Tool Calling으로 작업을 위임할 수 있도록 한다.

### 5. 핵심 정리

Supervisor는 **어떤 Agent에게 어떤 업무를 위임할지 결정**하고, 전문 Agent는 **자신의 도구를 사용하여 위임받은 작업을 처리**한다.

따라서 Supervisor 패턴은 Agent 간 작업 위임과 Agent 내부의 Tool Calling이 결합된 구조이다.

※ `langgraph-supervisor`는 기존 Supervisor 구조를 학습하는 데 유용하지만, 새로운 구현에서는 LangChain의 `create_agent()`를 활용해 전문 Agent를 도구로 제공하는 방식도 권장된다.

## 3. Supervisor 커스텀 프롬프트

### 1. Supervisor의 Custom Prompt란?

Supervisor의 `prompt`는 전문 Agent에게 어떤 작업을 위임할지, 어떤 순서로 작업을 진행할지 등 전체 업무 흐름에 대한 지침을 정의한다.

Custom Prompt를 통해 다음과 같은 사항을 지정할 수 있다.

- **작업 위임 기준:** 어떤 상황에서 어떤 Agent를 사용할지
- **작업 순서:** 어떤 Agent를 먼저 실행할지
- **결과 기준:** 최종 답변에 포함해야 할 내용
- **종료 지침:** 언제 최종 답변을 생성할지

### 2. Supervisor Prompt와 Agent System Prompt

| 구분 | Supervisor `prompt` | Agent `system_prompt` |
| --- | --- | --- |
| 역할 | 전체 작업 조율 | 개별 업무 수행 |
| 주요 내용 | 작업 분배 및 위임 순서 | Agent의 역할과 행동 지침 |
| 적용 대상 | Supervisor LLM | 해당 전문 Agent의 LLM |

### 3. Custom Prompt 구현

```
custom_prompt = (
    "당신은 팀 Supervisor입니다.\n"
    "1. 먼저 researcher에게 자료 조사를 요청하세요.\n"
    "2. 이후 summarizer에게 요약을 요청하세요.\n"
    "3. 확인된 출처가 있다면 최종 답변에 포함하세요.\n"
    "4. 작업이 끝나면 최종 답변을 생성하세요."
)

workflow = create_supervisor(
    agents=[researcher, summarizer],
    model=model,
    prompt=custom_prompt
)
```

`prompt=custom_prompt`를 통해 Supervisor의 위임 및 실행 지침을 지정한다.

### 4. Prompt 기반 제어와 Graph 기반 제어

**Prompt 기반 제어**

- LLM에 원하는 작업 순서와 행동 지침을 제공한다.
- LLM이 프롬프트를 참고해 다음 Agent를 결정한다.
- 유연하지만 실행 순서를 반드시 보장하지는 않는다.

**Graph 기반 제어**

```
graph.add_edge("researcher", "summarizer")
graph.add_edge("summarizer", END)
```

- 노드와 엣지를 직접 정의한다.
- 정의된 경로를 따라 실행하도록 구성할 수 있다.
- 엄격한 작업 순서를 구현하는 데 적합하다.

단, 작업의 성공 여부까지 검증해야 한다면 별도의 검증 로직이 필요하다.

### 핵심 정리

**Supervisor의 Custom Prompt는 LLM의 행동을 유도하는 지침이며, `StateGraph`의 Edge는 코드 수준에서 실행 경로를 정의한다.**

유연한 작업 위임이 중요하면 Prompt 기반 제어를, 반드시 지켜야 하는 실행 순서가 있다면 Graph 기반 제어와 검증 로직을 활용한다.

## 4. Swarm 패턴과 `create_swarm()`

### 1. Swarm 패턴이란?

Swarm은 중앙 Supervisor 없이 현재 활성화된 Agent가 다른 Agent에게 직접 작업과 제어권을 넘기는 멀티 에이전트 패턴이다.

**Supervisor와 Swarm 비교**

![스크린샷 2026-11-11 오전 3.43.52.png](%EC%9C%84%ED%82%A4%EB%8F%85%EC%8A%A4%2013/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-11-11_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_3.43.52.png)

| 구분 | Supervisor | Swarm |
| --- | --- | --- |
| Handoff 주체 | Supervisor | 현재 활성 Agent |
| Agent 전환 | Supervisor 경유 | Agent 간 직접 전환 |
| 핵심 특징 | 중앙 집중형 조율 | 자율적인 협업 |

핵심 차이 = 누가 다음 Agent를 결정하느냐

- `Supervisor`: 중앙 Supervisor가 결정
- `Swarm`: 현재 활성화된 Agent가 결정

### 2. `create_handoff_tool()`

```
transfer_to_tech = create_handoff_tool(
    agent_name="tech_agent",
    description="기술 문의일 때 tech_agent로 전환합니다."
)
```

Agent가 다른 Agent에게 제어권을 넘길 수 있도록 Handoff Tool을 생성한다.

- `agent_name`: 전환 대상 Agent
- `name`: Handoff Tool 이름 (선택 사항)
- `description`: Handoff할 상황에 대한 설명

**Handoff 내부 동작**

![스크린샷 2026-11-11 오전 3.43.31.png](%EC%9C%84%ED%82%A4%EB%8F%85%EC%8A%A4%2013/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-11-11_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_3.43.31.png)

1. 현재 Agent가 Handoff Tool 호출
2. 전환 확인을 위한 `ToolMessage` 생성
3. `Command`로 다음 실행 Agent 지정
4. 공유 State의 `active_agent` 갱신
5. 대상 Agent 실행

### 3. `active_agent`

`active_agent`는 현재 활성화된 Agent의 이름을 저장하는 State 필드이다.

```
{
    "active_agent": "tech_agent"
}
```

Handoff가 발생하면 대상 Agent의 이름으로 갱신된다.

체크포인터를 사용하면 다음 대화에서도 마지막 활성 Agent부터 이어서 실행할 수 있다.

### 4. `create_swarm()`

```
workflow = create_swarm(
    agents=[general_agent, tech_agent],
    default_active_agent="general_agent"
)
```

- `agents`: Swarm에 참여하는 Agent 목록
- `default_active_agent`: 처음 실행할 기본 Agent

각 Agent의 `tools`에 등록한 Handoff Tool이 전환 가능한 경로를 결정한다.

### 5. 체크포인터를 통한 상태 유지

```
checkpointer = InMemorySaver()

app = workflow.compile(
    checkpointer=checkpointer
)

config = {
    "configurable": {"thread_id": "user_001"}
}
```

- `InMemorySaver`: 그래프 상태를 메모리에 저장
- `checkpointer`: 여러 실행에 걸쳐 State를 유지
- `thread_id`: 대화 세션을 구분하는 식별자

동일한 `thread_id`로 실행하면 이전 메시지와 활성 Agent 상태를 이어서 사용할 수 있다.

단, `InMemorySaver`는 프로세스 종료 시 저장된 데이터가 사라지므로 개발 및 실습용으로 적합하다.

### 핵심 정리

**Swarm은 현재 Agent가 Handoff Tool을 통해 다른 Agent에게 직접 제어권을 넘기는 구조이다.**

- `create_handoff_tool()`: Agent 간 전환
- `create_swarm()`: Swarm 워크플로 구성
- `active_agent`: 현재 활성 Agent 관리
- `checkpointer`: 멀티 턴 대화의 상태 유지

## 5. Custom Handoff와 멀티 턴 상태 유지

### 1. Custom Handoff란?

Swarm 패턴에서는 현재 활성화된 Agent가 Handoff Tool을 호출하여 다른 Agent에게 제어권을 전달한다.

`create_handoff_tool()`의 이름과 설명을 커스터마이징하면 LLM이 어떤 상황에 어떤 Agent를 호출할지 판단하는 데 도움을 줄 수 있다.

```
transfer_to_specialist = create_handoff_tool(
    agent_name="specialist",
    name="escalate_to_specialist",
    description="복잡한 기술 문제를 specialist에게 전달합니다."
)
```

- `agent_name`: Handoff 대상 Agent의 이름
- `name`: LLM이 호출하는 Handoff Tool의 이름
- `description`: 도구를 사용해야 하는 상황에 대한 설명

**`agent_name`과 `name`은 반드시 같을 필요가 없다.**

### 2. Handoff Tool 등록과 실행

```
frontline = create_agent(
    model=model,
    tools=[diagnose_basic, transfer_to_specialist],
    name="frontline",
    system_prompt="일반 상담을 담당합니다."
)
```

현재 Agent의 `tools`에 Handoff Tool을 등록하면 LLM이 필요에 따라 해당 도구를 선택할 수 있다.

**실행 흐름**

1. 현재 Agent가 사용자 요청 분석
2. Handoff Tool 호출 결정
3. Handoff Tool이 `Command`를 통해 다음 Agent로 제어권 전달
4. `active_agent` 갱신
5. 대상 Agent 실행

단, `description`은 LLM의 선택을 유도할 뿐 특정 Handoff 조건을 반드시 보장하지는 않는다.

### 3. `active_agent`란?

Swarm에서 현재 활성화된 Agent를 나타내는 State 필드이다.

```
{
    "active_agent": "specialist"
}
```

Handoff가 발생하면 대상 Agent로 갱신되고, 다음 대화의 시작 Agent를 결정하는 데 활용될 수 있다.

### 4. Checkpointer란?

Checkpointer는 그래프의 실행 상태를 저장하고 복원하여 여러 대화 턴에 걸쳐 이전 State를 이어서 사용할 수 있게 하는 기능이다.

```
from langgraph.checkpoint.memory import InMemorySaver

app = workflow.compile(
    checkpointer=InMemorySaver()
)
```

체크포인터를 통해 메시지 내역, `active_agent` 등의 상태 정보를 유지할 수 있다.

### 5. `thread_id`를 통한 멀티 턴 상태 유지

```
config = {
    "configurable": {
        "thread_id": "user_001"
    }
}

result1 = app.invoke(
    {"messages": [{"role": "user", "content": "앱이 꺼져요."}]},
    config=config
)

result2 = app.invoke(
    {"messages": [{"role": "user", "content": "여전히 안 돼요."}]},
    config=config
)
```

- `thread_id`: 대화 세션의 식별자
- 동일한 `thread_id`: 해당 대화의 저장된 State를 이어서 사용
- 다른 `thread_id`: 다른 대화 상태로 구분

**이전 Agent가 다음 턴에서도 이어서 처리하려면 Checkpointer를 연결하고 같은 `thread_id`로 실행해야 한다.**

### 6. `InMemorySaver`의 한계

`InMemorySaver`는 메모리에 체크포인트를 저장하기 때문에 개발 및 실습에 적합하다.

프로세스가 종료되면 저장된 데이터가 사라지므로, 영구적인 저장이 필요한 서비스에서는 데이터베이스 기반 체크포인터를 고려해야 한다.

### 핵심 정리

- **Custom Handoff:** Agent 간 전환 도구의 이름과 설명을 조정
- **`active_agent`:** 현재 활성 Agent를 나타내는 State 정보
- **Checkpointer:** 이전 실행 상태 저장 및 복원
- **`thread_id`:** 어떤 대화의 상태를 이어갈지 식별

Swarm에서 Handoff는 Agent의 제어권을 변경하고, Checkpointer는 그 상태가 다음 실행에서도 유지되도록 한다.

## 6. Hierarchical 패턴

### 1. Hierarchical 패턴이란?

![스크린샷 2026-11-11 오전 4.53.34.png](%EC%9C%84%ED%82%A4%EB%8F%85%EC%8A%A4%2013/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-11-11_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_4.53.34.png)

**Hierarchical 패턴**은 여러 Supervisor를 계층적으로 구성하여 복잡한 멀티 에이전트 시스템을 팀 단위로 관리하는 아키텍처이다.

일반적인 Supervisor 패턴에서는 하나의 Supervisor가 여러 전문 Agent를 관리하지만, Hierarchical 패턴에서는 상위 Supervisor가 하위 Supervisor 또는 팀 그래프를 관리한다.

**실행 구조**

```
Top Supervisor
    ├── Research Team
    │     ├── Research Supervisor
    │     ├── Searcher
    │     └── Analyst
    │
    └── Writing Team
          ├── Writing Supervisor
          ├── Drafter
          └── Editor
```

### 2. Hierarchical 패턴을 사용하는 이유

하나의 Supervisor가 모든 전문 Agent를 직접 관리하면 Agent가 많아질수록 관리 복잡도가 증가하고, 중앙 Supervisor의 의사결정 과정이 병목이 될 수 있다.

계층적 구조에서는 역할을 팀 단위로 나누어 다음과 같이 관리한다.

- **Top Supervisor:** 팀 단위 작업 위임과 전체 업무 조율
- **Team Supervisor:** 팀 내부의 전문 Agent 관리 및 작업 위임
- **Worker Agent:** 실제 전문 작업 수행

각 팀을 Subgraph로 구성하면 독립적인 개발과 테스트가 가능하며, 상위 Supervisor는 팀 내부의 세부 작업까지 직접 관리할 필요가 없다.

### 3. 팀 단위 Supervisor 구현

```
research_team = create_supervisor(
    agents=[searcher, analyst],
    model=model,
    supervisor_name="research_lead",
    output_mode="last_message"
)

research_app = research_team.compile(
    name="research_team"
)
```

- `supervisor_name`: 팀 내부 Supervisor 노드의 이름
- `compile(name=...)`: 컴파일된 팀 그래프의 이름
- `output_mode`: 관리 대상 Agent의 메시지 기록 반영 방식

### 4. Top Supervisor 구현

```
top_supervisor = create_supervisor(
    agents=[research_app, writing_app],
    model=model,
    prompt="리서치 팀과 작성 팀의 작업을 조율하세요.",
    output_mode="full_history"
)

app = top_supervisor.compile()
```

일반적인 Supervisor 패턴과 달리 `agents`에 개별 전문 Agent가 아닌 **컴파일된 팀 그래프**를 전달한다.

### 5. `output_mode`의 역할

| 구분 | `last_message` | `full_history` |
| --- | --- | --- |
| 반영 범위 | 관리 대상 Agent의 마지막 메시지 | 관리 대상 Agent의 전체 메시지 이력 |
| 장점 | 메시지 양 감소 | 자세한 맥락 유지 |
| 단점 | 세부 정보 누락 가능 | 불필요한 메시지 증가 가능 |

계층적 구조에서는 하위 팀의 핵심 결과만 상위로 전달하도록 `last_message`를 사용하는 설계를 고려할 수 있다.

다만 마지막 메시지에 필요한 정보가 포함되어 있어야 한다.

### 핵심 정리

**Hierarchical 패턴의 핵심은 Supervisor가 개별 Agent가 아닌 팀 단위 그래프를 관리하도록 구성하는 것이다.**

- 관리 대상 Agent의 수와 복잡도를 계층적으로 분산
- 팀별 독립적인 역할과 책임 정의
- Subgraph를 활용한 모듈화
- `output_mode`를 활용한 메시지 기록 범위 조절

단, 계층 구조가 복잡해질수록 LLM 호출 횟수와 지연 시간이 증가할 수 있으므로 시스템 규모와 요구사항을 고려해야 한다.

## 7. State 공유와 Message Passing

### 1. Message Passing이란?

![스크린샷 2026-11-11 오전 4.53.22.png](%EC%9C%84%ED%82%A4%EB%8F%85%EC%8A%A4%2013/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-11-11_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_4.53.22.png)

**Message Passing**은 여러 Agent가 메시지를 통해 작업 결과와 맥락을 주고받는 방식이다.

Multi-Agent 시스템에서는 이전 Agent가 수행한 작업 결과를 다른 Agent가 활용해야 하므로 Agent 간 정보 전달이 필요하다.

예를 들어 Researcher가 조사한 내용을 Writer가 보고서로 작성하려면 조사 결과가 Writer에게 전달되어야 한다.

### 2. 공유 State와 `MessagesState`

```
from langgraph.graph import MessagesState

class TeamState(MessagesState):
    report: str
```

- `messages`: 대화 및 Agent·Tool 실행 메시지 관리
- `report`: 별도로 관리하는 보고서 데이터

`MessagesState`에는 `messages` 필드와 `add_messages` 병합 규칙이 정의되어 있다.

### 3. `output_mode`를 통한 메시지 전달 범위 설정

```
workflow = create_supervisor(
    agents=[researcher, writer],
    model=model,
    output_mode="full_history"
)
```

| 모드 | 동작 | 특징 |
| --- | --- | --- |
| `last_message` | 전문 Agent의 마지막 메시지 반영 | 메시지 양 감소, 세부 정보 누락 가능 |
| `full_history` | 전문 Agent의 전체 메시지 이력 반영 | 자세한 맥락 유지, 메시지 양 증가 |

`last_message`를 사용하더라도 마지막 메시지에 필요한 작업 결과가 충분히 포함되어 있다면 다음 Agent가 작업을 이어갈 수 있다.

반면 마지막 메시지에 `"조사 완료"` 같은 간단한 정보만 있다면 작업 결과를 파악하기 어려울 수 있다.

### 4. 공유 메시지 확인

```
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Python에 대해 조사하고 정리해줘."
        }
    ]
})

for msg in result["messages"]:
    print(msg.type, getattr(msg, "name", None), msg.content)
```

- `msg.type`: 메시지 종류
- `msg.name`: 메시지에 지정된 이름(존재하는 경우)
- `msg.content`: 메시지 내용

이를 통해 Agent 간 작업 과정에서 생성된 메시지 기록을 확인할 수 있다.

### 5. State 관리 관련 핵심 기능 비교

| 기능 | 역할 |
| --- | --- |
| `output_mode` | 전문 Agent의 메시지 중 공유할 범위를 결정 |
| `add_messages` | 새로운 메시지를 기존 State의 메시지와 병합 |
| Checkpointer | State를 저장하고 이후 실행에서 복원 |
- `output_mode`는 메시지 **전달 범위**를 제어
- `add_messages`는 메시지 **병합 규칙**을 정의
- Checkpointer는 실행 간 State의 **저장과 복원**을 담당

### 핵심 정리

Multi-Agent 시스템에서는 각 Agent가 독립적인 작업을 수행하면서도 필요한 정보를 공유해야 한다.

**Message Passing은 Agent 간 협업을 연결하고, State는 공유 정보를 관리하며, Checkpointer는 실행 간 상태를 유지한다.**

메시지 전달 방식을 설계할 때는 불필요한 이력 전달을 줄이면서도 다음 Agent에게 필요한 정보가 누락되지 않도록 해야 한다.

## 8. 종합 실습 및 패턴 비교

### 1. 종합 실습: 리서치 리포트 생성 시스템

Multi-Agent System을 활용해 자료 조사, 보고서 작성, 검수를 수행하는 시스템을 구성할 수 있다.

![스크린샷 2026-11-11 오전 5.08.18.png](%EC%9C%84%ED%82%A4%EB%8F%85%EC%8A%A4%2013/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-11-11_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_5.08.18.png)

**Agent 역할**

| Agent | 역할 | 도구 |
| --- | --- | --- |
| Researcher | 자료 조사 | `search_web` |
| Writer | 보고서 작성 | `write_report` |
| Reviewer | 보고서 검수 | `review_report` |

**기본 실행 흐름**

```
User
  ↓
Supervisor
  ↓
Researcher → 자료 조사
  ↓
Supervisor
  ↓
Writer → 보고서 작성
  ↓
Supervisor
  ↓
Reviewer → 검수
  ↓
Supervisor → 최종 응답
```

이는 Supervisor가 조사, 작성, 검수 순서로 Agent를 위임한 경우의 예시이며, 실제 순서는 LLM의 판단에 따라 달라질 수 있다.

### 2. 종합 실습 코드

```
workflow = create_supervisor(
    agents=[researcher, writer, reviewer],
    model=model,
    prompt=(
        "먼저 researcher에게 자료 조사를 위임하고, "
        "writer에게 보고서 작성을, "
        "reviewer에게 검수를 요청하세요."
    ),
    output_mode="full_history"
)

app = workflow.compile(
    checkpointer=InMemorySaver()
)
```

- `agents`: Supervisor가 관리할 전문 Agent
- `model`: Supervisor의 의사결정에 사용하는 LLM
- `prompt`: 업무 분배 및 실행 지침
- `output_mode`: 전문 Agent의 메시지 공유 범위
- `checkpointer`: 실행 간 State 저장 및 복원

※ 이 코드는 앞서 Agent와 model 등이 정의되었다는 가정하의 핵심 부분이다.

### 3. 작업 순서 제어

Supervisor의 프롬프트로 작업 순서를 안내할 수 있지만, LLM이 반드시 해당 순서를 지킨다고 보장할 수는 없다.

엄격한 실행 순서가 필요하다면 `StateGraph`를 활용할 수 있다.

```
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")
graph.add_edge("reviewer", END)
```

고정된 엣지는 노드 실행 순서를 정의하지만, 각 작업의 성공 여부나 결과의 정확성을 보장하지는 않으므로 별도의 검증 로직이 필요할 수 있다.

### 4. Multi-Agent 아키텍처 비교

| 구분 | Supervisor | Swarm | Hierarchical |
| --- | --- | --- | --- |
| 제어 방식 | 중앙 집중 | 분산 | 계층적 |
| 다음 Agent 결정 | Supervisor | 현재 Agent | 계층별 Supervisor |
| 핵심 메커니즘 | 중앙 작업 위임 | 직접 Handoff | 팀 단위 업무 위임 |
| 대표 함수 | `create_supervisor()` | `create_swarm()` | 중첩 Supervisor |
| 활용 예시 | 보고서 작성 | 고객 상담 | 대규모 프로젝트 관리 |

**Supervisor**

- 중앙 Supervisor가 여러 전문 Agent를 관리한다.
- 작업 분배와 결과 취합을 중앙에서 수행한다.
- 중앙 의사결정에 따른 병목 가능성이 있다.

**Swarm**

- 현재 활성 Agent가 다른 Agent에게 직접 제어권을 전달한다.
- `create_handoff_tool()`을 사용하여 Handoff Tool을 생성한다.
- Agent 간 직접 전환이 필요한 시스템에 활용할 수 있다.

**Hierarchical**

- 상위 Supervisor가 여러 하위 Supervisor 또는 팀 그래프를 관리한다.
- 팀별 역할과 책임을 분리할 수 있다.
- 계층 증가에 따른 비용과 복잡도를 고려해야 한다.

### 5. Agent 간 State와 메시지 관리

- Handoff Tool : Agent 간 실행 제어권 전달
- `active_agent` : 현재 활성화된 Agent 관리
- `output_mode` : 전문 Agent의 메시지 공유 범위 결정
- `add_messages` : State의 메시지 병합
- Checkpointer : 실행 간 State 저장 및 복원
- `thread_id` : 대화 세션 식별

특히 `output_mode="last_message"`를 사용한다면 마지막 메시지에 다음 Agent가 필요로 하는 작업 결과가 충분히 포함되어야 한다.

### 6. 아키텍처 선택 기준

- **Single-Agent:** 하나의 Agent와 여러 도구만으로 해결 가능한 작업
- **Supervisor:** 여러 전문 Agent를 중앙에서 조율해야 하는 작업
- **Swarm:** 현재 Agent가 다른 Agent에게 직접 업무를 넘기는 작업
- **Hierarchical:** 여러 전문 팀을 계층적으로 관리해야 하는 작업

Multi-Agent를 사용한다고 항상 성능이 향상되는 것은 아니다.

**작업의 복잡도, Agent 간 의존성, 실행 경로의 유연성, 비용과 지연 시간**을 고려하여 적절한 구조를 선택해야 한다.

### 핵심 정리

Multi-Agent System의 핵심은 여러 Agent를 만드는 것 자체가 아니라 **역할을 적절히 분담하고, Agent 간 작업 위임과 정보 전달 방식을 설계하는 것**이다.

- Supervisor는 중앙에서 Agent를 선택한다.
- Swarm은 현재 Agent가 직접 Handoff한다.
- Hierarchical은 팀 단위로 계층적인 작업 위임을 수행한다.

이러한 구조를 활용하면 복잡한 업무를 여러 전문 Agent가 협력하여 처리하도록 구성할 수 있다.