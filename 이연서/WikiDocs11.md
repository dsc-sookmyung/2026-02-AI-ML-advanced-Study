# 위키독스 11

개념: LangGraph 에이전트

## 1. ReAct 패턴이란?

**ReAct(Reasoning + Acting):** LLM이 추론(Thought) → 행동(Action) → 관찰(Observation)을 반복하며 문제를 해결하는 에이전트 패턴

![스크린샷 2026-11-06 오전 2.23.22.png](%EC%9C%84%ED%82%A4%EB%8F%85%EC%8A%A4%2011/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-11-06_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_2.23.22.png)

- **`Thought`:** LLM이 현재 정보를 바탕으로 다음 행동을 결정
- **`Action`:** 필요한 외부 도구를 호출하고 실행
- **`Observation`:** 도구 실행 결과를 확인하고 다시 LLM에 전달
- **`Final Answer`:** 더 이상 도구가 필요하지 않으면 최종 답변 생성

## 2. Tool Calling과 ReAct의 차이

- **`Tool Calling`:** LLM이 외부 도구의 실행을 요청할 수 있도록 하는 기능
- **`ReAct`:** 도구 호출과 그 결과를 바탕으로 반복적인 의사결정을 수행하는 패턴

ReAct는 Tool Calling을 활용할 수 있으며, 단순히 도구를 호출하는 것뿐 아니라 **언제 어떤 도구를 사용할지, 언제 종료할지를 LLM이 결정한다**는 점이 핵심

## 3. LangGraph에서의 실행 흐름

`User → Agent → ToolNode → Agent → ... → END`

1. `Agent`: LLM이 도구 호출 여부를 결정
2. `AIMessage.tool_calls`: LLM이 요청한 도구 정보 저장
3. `ToolNode`: 요청된 도구 실행
4. `ToolMessage`: 실행 결과를 메시지에 추가
5. `Agent`: 결과를 바탕으로 다음 행동 결정
6. 도구 호출이 더 이상 없으면 종료

결국, ReAct는 LLM의 추론 능력과 외부 도구의 실행 능력을 결합하여 복잡한 문제를 단계적으로 해결

**핵심) 도구 호출 → 결과 관찰 → 다음 행동 결정**이 반복된다

## 4. `create_react_agent`

### 1. 개념

`create_react_agent()`: ReAct 패턴을 구현한 에이전트 그래프를 자동으로 생성하는 함수

⇒ 즉, ReAct 에이전트 구현에 필요한 그래프 구성을 자동화

자동화하는 작업 (`StateGraph`를 직접 구성시 필요한 작업들)

![스크린샷 2026-11-07 오후 3.39.10.png](%EC%9C%84%ED%82%A4%EB%8F%85%EC%8A%A4%2011/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-11-07_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.39.10.png)

- **Agent 노드 생성:** LLM에 도구를 연결하고 추론 수행
- **Tools 노드 생성:** `ToolNode`를 통해 도구 실행
- **조건부 분기:** `tools_condition`으로 도구 호출 여부 확인
- **순환 엣지:** 도구 실행 결과를 다시 LLM에 전달

### 2. 핵심 구성 요소

- `LLM` : 어떤 도구를 호출할지 선택
- `bind_tools()` : LLM에 사용 가능한 도구 정보 제공
- `AIMessage.tool_calls` : LLM이 요청한 도구 호출 정보
- `tools_condition` : 도구 호출 여부에 따라 이동 경로 선택
- `ToolNode` : 요청된 도구 실행
- `ToolMessage` : 도구 실행 결과 저장

### 3. 도구 선택 과정

```python

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def multiply(a: int, b: int) -> int:
    """두 정수의 곱을 계산합니다."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """두 정수의 합을 계산합니다."""
    return a + b

model = ChatOpenAI(model="gpt-5-mini")

agent = create_react_agent(model, [multiply, add])

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "23 곱하기 17은?"}
    ]
})

```

1. LLM이 사용자 질문과 사용 가능한 도구 정보를 확인한다.
2. 도구 이름, 설명(`docstring`), 매개변수 정보를 바탕으로 사용할 도구를 결정한다.
3. 선택한 도구와 인자 정보를 `AIMessage.tool_calls`에 담는다.
4. `tools_condition`이 도구 호출 여부를 확인한다.
5. `ToolNode`가 요청된 도구를 실행한다.
6. 결과를 `ToolMessage`에 담아 다시 LLM에 전달한다.

### 4. 핵심 정리

**LLM은 도구를 선택하고, `tools_condition`은 실행 경로를 결정하며, `ToolNode`는 도구를 실행**

※ LangGraph v1에서 `create_react_agent()`는 deprecated 상태이며, 현재는 `langchain.agents.create_agent()` 사용을 권장

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[multiply, add],
    system_prompt="계산이 필요하면 제공된 도구를 사용하세요."
)
```

## 5. **`StateGraph`를 활용한 커스텀 ReAct**

### 1. `StateGraph`를 사용하는 이유

`create_react_agent()`는 ReAct 패턴에 필요한 그래프를 자동으로 생성하지만, 실행 흐름을 세밀하게 제어해야 하는 경우에는 `StateGraph`를 활용해 직접 구현할 수 있다.

예를 들어 도구 호출 전 권한 검증, 실행 횟수 제한, 조건에 따른 분기 등을 구현할 수 있다.

### 2. 커스텀 ReAct의 구성 요소

- `State` : 노드 간에 공유하는 데이터 관리
- `Agent Node` : LLM 호출 및 응답 생성
- `Tools Node` : LLM이 요청한 도구 실행
- `Conditional Edges` : 조건에 따라 다음 노드 선택

### 3. State란?

`State`: 그래프의 여러 노드가 공유하는 데이터 구조

사용자 메시지, LLM 응답, 도구 실행 결과 등 그래프 실행에 필요한 정보를 관리

```python
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]
```

- `TypedDict`: State의 키와 값의 타입 정의
- `messages`: 메시지 내역을 관리하는 필드
- `BaseMessage`: 다양한 메시지 유형의 기반 클래스
- `Annotated`: 필드에 추가적인 메타데이터 지정
- `add_messages`: 기존 메시지와 새로운 메시지를 병합하는 함수

**+ `add_messages`를 사용하는 이유**

일반적인 State 업데이트에서는 기존 필드 값이 덮어써질 수 있다.

`add_messages`를 지정하면 기존 메시지를 유지하면서 새 메시지를 추가할 수 있으며, 동일한 ID의 메시지는 업데이트할 수 있다.

예시:

```
[HumanMessage]
       ↓
[HumanMessage, AIMessage]
       ↓
[HumanMessage, AIMessage, ToolMessage]
       ↓
[HumanMessage, AIMessage, ToolMessage, AIMessage]
```

※ State는 그래프 실행 중 데이터를 관리하지만, 여러 실행에 걸쳐 대화 상태를 유지하려면 체크포인터 등 별도의 지속성 기능이 필요하다.

### 4. `StateGraph` 구현

```python
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

model_with_tools = model.bind_tools([multiply])

def agent_node(state: AgentState):
    response = model_with_tools.invoke(
        state["messages"]
    )
    return {"messages": [response]}

graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode([multiply]))

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    tools_condition
)

graph.add_edge("tools", "agent")

app = graph.compile()
```

**핵심 함수**

- `StateGraph(AgentState)`: State 구조를 기반으로 그래프 생성
- `add_node()`: 실행할 노드 등록
- `set_entry_point()`: 시작 노드 지정
- `add_conditional_edges()`: 조건에 따라 다음 이동 경로 결정
- `add_edge()`: 노드 간 연결 정의
- `compile()`: 정의한 그래프를 실행 가능한 형태로 컴파일

### 5. 실행 흐름

```
START
  ↓
Agent
  ↓
tools_condition
  ├── tool_calls 있음 → ToolNode
  │                       ↓
  │                     Agent
  │                       ↓
  │                    (반복)
  │
  └── tool_calls 없음 → END
```

`graph.add_edge("tools", "agent")`가 중요한 이유는 도구 실행 이후 다시 LLM이 다음 행동을 판단하도록 만들기 때문이다.

이 연결을 통해 ReAct의 **추론 → 행동 → 관찰 → 재추론** 순환 구조를 구현한다.

### 핵심 정리

`StateGraph`는 노드와 엣지를 직접 정의하여 에이전트의 실행 흐름을 제어할 수 있도록 한다.

- **`State`:** 노드 간 실행 맥락 공유
- **`Node`:** 실제 작업 수행
- **`Edge`:** 노드 간 이동 경로 정의
- **`Conditional Edge`:** 실행 결과에 따라 이동 경로 결정

`create_react_agent()`가 ReAct 그래프를 자동으로 구성하는 방식이라면, `StateGraph`는 개발자가 직접 실행 구조를 설계할 수 있는 방식

![스크린샷 2026-11-11 오전 2.33.58.png](%EC%9C%84%ED%82%A4%EB%8F%85%EC%8A%A4%2011/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-11-11_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_2.33.58.png)

### 6. ReAct 반복 횟수 제한 커스텀

#### 1. 반복 횟수를 제한해야 하는 이유

ReAct 에이전트는 LLM이 계속 도구 호출을 요청하면 `Agent → Tools → Agent` 순환 구조가 반복될 수 있다.

이로 인해 불필요한 API 비용 증가, 응답 지연, 반복적인 도구 실행 등이 발생할 수 있으므로 최대 반복 횟수를 제한할 필요가 있다.

#### 2. State에 반복 횟수 추가

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int

MAX_ITERATIONS = 5
```

- `messages`: 기존 대화 및 도구 실행 메시지 관리
- `iteration_count`: Agent 노드의 실행 횟수 관리
- `MAX_ITERATIONS`: 허용할 최대 Agent 실행 횟수

#### 3. Agent 실행 횟수 업데이트

```python
def agent_node(state: AgentState):
    count = state.get("iteration_count", 0) + 1

    response = model_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response],
        "iteration_count": count
    }
```

![스크린샷 2026-11-11 오전 2.33.06.png](%EC%9C%84%ED%82%A4%EB%8F%85%EC%8A%A4%2011/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2026-11-11_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_2.33.06.png)

Agent가 실행될 때마다 `iteration_count`를 1 증가시키고, State 업데이트 값으로 반환한다.

`messages`는 `add_messages`를 통해 병합되며, 별도의 Reducer가 지정되지 않은 `iteration_count`는 새로운 값으로 갱신된다.

#### 4. 조건부 분기 함수 정의

```python
def should_continue(state: AgentState):
    last_msg = state["messages"][-1]

    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "end"

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    return "end"
```

**조건 검사 순서**

1. 최대 실행 횟수에 도달했다면 `"end"` 반환
2. 실행 횟수 제한에 도달하지 않았고 `tool_calls`가 있다면 `"tools"` 반환
3. 추가 도구 호출이 필요하지 않으면 `"end"` 반환

#### 5. 조건부 엣지 연결

```python
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

graph.add_edge("tools", "agent")
```

`should_continue()`의 반환값에 따라 `tools` 노드로 이동하거나 그래프를 종료한다.

`tools → agent` 연결은 유지되므로 ReAct의 순환 구조는 동일하다.

#### 핵심 정리

- `tools_condition`: 도구 호출 여부에 따른 기본 분기
- `should_continue`: 반복 횟수 등 추가 조건을 반영한 사용자 정의 분기
- `iteration_count`: 이번 예제에서는 Agent 노드 실행 횟수
- **실행 제한에 도달하면 도구 호출 요청이 있어도 종료될 수 있다.**

※ 최종 답변 생성 전에 실행이 종료될 가능성이 있으므로 실제 서비스에서는 종료 안내나 별도 처리 로직이 필요하다.

## 6. Streaming과 중간 단계 관찰

### 1. Streaming이 필요한 이유

ReAct 에이전트는 LLM 호출과 도구 실행을 반복하므로 최종 답변을 생성하기까지 시간이 걸릴 수 있다.

`stream()`을 활용하면 실행이 완전히 종료되기 전에 중간 결과나 상태 변화를 실시간으로 확인할 수 있다.

**`invoke()`와 `stream()` 비교**

| 구분 | `invoke()` | `stream()` |
| --- | --- | --- |
| 반환 방식 | 실행 완료 후 결과 반환 | 실행 도중 데이터를 나누어 전달 |
| 중간 과정 확인 | 기본 반환값만으로는 어려움 | 가능 |
| 활용 | 최종 결과 처리 | 실시간 상태 관찰 및 출력 |

### 2. `stream_mode` 종류

| 모드 | 반환 데이터 | 활용 목적 |
| --- | --- | --- |
| `values` | 각 단계의 전체 State | 전체 실행 상태 확인 |
| `updates` | 각 단계에서 발생한 State 변경분 | 노드별 실행 결과 확인 |
| `messages` | LLM의 메시지 조각 및 메타데이터 | 실시간 답변 출력 |

### 3. 코드 예시

**① `updates`: 노드별 변경 사항 확인**

```python
for chunk in agent.stream(
    input_msg,
    stream_mode="updates"
):
    print(chunk)
```

**② `values`: 전체 State 확인**

```python
for state in agent.stream(
    input_msg,
    stream_mode="values"
):
    print(state["messages"])
```

**③ `messages`: LLM의 메시지 조각 수신**

```python
for msg, metadata in agent.stream(
    input_msg,
    stream_mode="messages"
):
    if msg.content:
        print(msg.content, end="", flush=True)
```

※ 위 코드는 기본 스트리밍 출력 형식을 기준으로 한다. 사용하는 LangGraph 버전과 스트리밍 API 옵션에 따라 반환 데이터 구조는 달라질 수 있다.

### 4. 핵심 정리

- **`values`**: 현재까지의 전체 State 확인
- **`updates`**: 노드 실행으로 변경된 데이터 확인
- **`messages`**: LLM이 생성 중인 메시지 조각 수신

`stream()`은 에이전트 실행 중의 데이터를 전달받을 수 있으므로 실시간 챗봇 UI, 실행 과정 디버깅, 중간 상태 모니터링 등에 유용하다.

## 7. `astream_events()`를 활용한 이벤트 추적

### 1. `astream_events()`란?

`astream_events()`는 에이전트의 실행 과정에서 발생하는 세부 이벤트를 비동기적으로 스트리밍하는 메서드이다.

`stream()`이 주로 State 변화와 메시지 스트리밍에 활용된다면, `astream_events()`는 LLM 호출이나 도구 실행 등의 시작·진행·종료 이벤트를 세밀하게 추적하는 데 활용할 수 있다.

### 2. 주요 이벤트 종류

| 이벤트 | 의미 |
| --- | --- |
| `on_chat_model_start` | 채팅 모델 실행 시작 |
| `on_chat_model_stream` | LLM 메시지 조각 생성 |
| `on_chat_model_end` | 채팅 모델 실행 완료 |
| `on_tool_start` | 도구 실행 시작 |
| `on_tool_end` | 도구 실행 완료 |
| `on_tool_error` | 도구 실행 중 오류 발생 |
| `on_chain_start` | 체인 또는 그래프 구성 요소 실행 시작 |
| `on_chain_end` | 체인 또는 그래프 구성 요소 실행 완료 |

### 3. 코드 예시

```
async for event in agent.astream_events(
    input_msg,
    version="v2"
):
    kind = event["event"]

    if kind == "on_tool_start":
        print("도구 시작:", event["name"])

    elif kind == "on_tool_end":
        print("실행 결과:", event["data"].get("output"))

    elif kind == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if isinstance(chunk.content, str) and chunk.content:
            print(chunk.content, end="", flush=True)
```

- `async for`: 비동기적으로 전달되는 이벤트를 순차적으로 처리
- `event["event"]`: 이벤트 종류 확인
- `event["name"]`: 이벤트가 발생한 Runnable 또는 도구 이름
- `event["data"]`: 이벤트의 입력·출력 등 관련 정보
- `run_id`: 개별 실행을 식별하는 ID
- `parent_ids`: 상위 실행의 ID 목록

### 4. `should_continue()`와의 차이

**`should_continue()`**

- 에이전트의 실행 상태를 검사
- 다음에 실행할 노드 또는 종료 경로 결정
- 그래프의 실행 흐름을 제어

**`astream_events()` 내부의 `if/elif`**

- 이미 발생한 이벤트의 종류 확인
- 조건에 맞게 이벤트 정보 출력 또는 처리
- 그래프의 실행 경로를 직접 변경하지 않음

### 5. 핵심 정리

- `stream()`: State 변경 사항이나 메시지 등 실행 중인 데이터 수신
- `astream_events()`: LLM 및 도구 실행 과정에서 발생하는 이벤트 추적
- `on_tool_start`: 도구 실행 시작 시점
- `on_tool_end`: 도구 실행 완료 시점과 정상 실행 결과 확인

`astream_events()`는 에이전트의 실행 과정을 상세하게 관찰할 수 있어 디버깅, 로깅, 모니터링에 유용하다.

## 8. 오류 처리와 재시도 전략

### 1. 에이전트 오류 처리가 필요한 이유

ReAct 에이전트는 LLM과 외부 도구를 반복적으로 호출하므로 다음과 같은 오류가 발생할 수 있다.

- 외부 API 연결 실패 또는 타임아웃
- LLM이 잘못된 도구 인자를 생성
- 도구 실행 중 예외 발생
- 에이전트의 과도한 반복 실행

오류를 적절히 처리하지 않으면 에이전트 실행이 중단되거나 정상적인 답변을 제공하지 못할 수 있다.

### 2. `handle_tool_errors` — 도구 오류 처리

```
graph.add_node(
    "tools",
    ToolNode([divide], handle_tool_errors=True)
)
```

`handle_tool_errors=True`를 설정하면 일반적인 도구 실행 오류를 잡아서 오류 내용을 `ToolMessage`로 반환할 수 있다.

**실행 흐름**

1. LLM이 `AIMessage.tool_calls`로 도구 실행 요청
2. `ToolNode`가 도구 실행
3. 실행 중 예외 발생
4. 예외 정보를 `ToolMessage`로 변환
5. State의 `messages`에 반영
6. Agent가 다시 실행되면서 오류 메시지를 참고해 다음 행동 결정

**주의:** `handle_tool_errors=True`는 오류 처리 기능이지 자동 재시도 기능이 아니다.

LLM이 오류를 보고 도구를 다시 호출할 수 있지만, 반드시 재시도하는 것은 아니다.

### 3. Retry — 작업 재시도

Retry는 작업 실행이 실패했을 때 미리 정의한 정책에 따라 다시 실행하는 전략이다.

LangGraph에서는 `RetryPolicy`를 사용할 수 있다.

```
from langgraph.types import RetryPolicy

graph.add_node(
    "search",
    search_node,
    retry_policy=RetryPolicy(
        max_attempts=3,
        retry_on=ConnectionError
    )
)
```

- `max_attempts=3`: 최초 실행을 포함하여 최대 3회 시도
- `retry_on=ConnectionError`: 지정한 예외에 대해 재시도

**Error Handling과 Retry의 차이**

- Error Handling: 발생한 오류를 처리
- Retry: 실패한 작업을 다시 실행

오류를 이미 정상적인 반환값으로 변환했다면 해당 예외에 대한 자동 재시도가 실행되지 않을 수 있다.

### 4. `iteration_count`와 `recursion_limit`

두 기능 모두 과도한 반복 실행을 제한하지만 관리 대상이 다르다.

| 구분 | `iteration_count` | `recursion_limit` |
| --- | --- | --- |
| 구현 | 개발자가 State에 정의 | LangGraph 실행 설정 |
| 측정 대상 | 직접 정의한 실행 횟수 | 그래프 실행 단계 수 |
| 제한 처리 | 사용자 정의 분기 | `GraphRecursionError` 발생 가능 |

**`iteration_count`**

```
count = state.get("iteration_count", 0) + 1
```

이번 예제에서는 Agent 노드가 실행될 때마다 1씩 증가시키도록 구현했다.

**`recursion_limit`**

```
result = agent.invoke(
    input_msg,
    config={"recursion_limit": 10}
)
```

그래프가 허용된 실행 단계 수를 초과하지 않도록 제한한다.

※ 그래프 실행 단계 수는 Agent 호출 횟수와 동일하지 않으므로 `recursion_limit=10`이 Agent의 정확한 10회 실행을 의미하지 않는다.

### 5. Fallback — 실패 시 대체 동작

Fallback은 정상적인 처리가 실패했을 때 미리 정의한 대체 행동을 수행하는 전략이다.

```
from langgraph.errors import GraphRecursionError

try:
    result = agent.invoke(
        input_msg,
        config={"recursion_limit": 10}
    )
except GraphRecursionError:
    result = (
        "반복 제한에 도달했습니다. "
        "질문을 다시 작성해 주세요."
    )
```

- `try`: 정상적인 에이전트 실행 시도
- `except GraphRecursionError`: 해당 예외 발생 시 처리
- Fallback: 사용자에게 대체 안내 메시지 제공

Fallback을 사용하는 이유는 실패 상황에서도 시스템이 예측 가능한 방식으로 대응하도록 하여 안정성과 사용자 경험을 개선하기 위해서이다.

### 6. 핵심 정리

| 전략 | 핵심 역할 |
| --- | --- |
| Error Handling | 오류를 처리하고 필요한 정보 전달 |
| Retry | 실패한 작업 재실행 |
| Iteration Limit | 개발자가 정의한 반복 횟수 제한 |
| Recursion Limit | 그래프 실행 단계 수 제한 |
| Fallback | 실패 시 대체 동작 제공 |

견고한 ReAct 에이전트를 구현하려면 **오류 처리 + 재시도 정책 + 반복 제한 + Fallback**을 상황에 맞게 조합하는 것이 중요하다.