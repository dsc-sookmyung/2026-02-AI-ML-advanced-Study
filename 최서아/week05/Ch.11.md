# Ch.11 LangGraph 에이전트: ReAct 패턴 구현

> 참고: https://wikidocs.net/333795  

## 1. 이번 챕터의 핵심

**ReAct = Reasoning(판단) + Acting(행동)**

- LLM이 질문을 분석하여 필요한 도구를 선택한다.
- 도구 실행 결과를 다시 LLM이 확인한다.
- 필요하면 다른 도구를 호출하거나 같은 도구를 재호출한다.
- 더 이상 도구가 필요 없으면 최종 답변을 생성한다.
- 중요한 점: **LLM은 도구 실행을 요청하며, 실제 실행은 외부 런타임이 담당한다.**

```text
사용자 질문
    ↓
Agent (LLM)
    ↓
도구 호출이 필요한가?
    ├─ 아니오 → 최종 응답 → END
    └─ 예 → Tools 실행
                ↓
          실행 결과 저장
                ↓
          Agent 재판단 ── (반복)
```

## 2. 이전 챕터와 연결

| 챕터 | 배운 내용 | ReAct에 활용되는 방식 |
|---|---|---|
| Ch.05 | Tool Calling | LLM이 도구와 인자를 선택 |
| Ch.07 | 임베딩·벡터 스토어 | 의미 기반 검색 도구 구성 |
| Ch.08 | RAG | 외부 문서를 검색해 답변에 활용 |
| Ch.09 | 메모리·대화 관리 | 이전 대화와 도구 결과 활용 |
| Ch.10 | LangGraph | State·Node·Edge로 반복 흐름 설계 |
| **Ch.11** | **ReAct** | **LLM 판단과 도구 실행을 순환 구조로 연결** |

## 3. 기본 Tool Calling과 ReAct의 차이

### 기본 Tool Calling

```python
model_with_tools = model.bind_tools([multiply])
response = model_with_tools.invoke("3과 5를 곱해줘")
print(response.tool_calls)
```

- `bind_tools()`는 사용 가능한 도구의 명세를 LLM에 제공한다.
- `tool_calls`는 **도구 호출 요청**이지 실행 결과가 아니다.
- 실제 함수 실행과 결과 재전달은 개발자 코드 또는 실행기가 담당한다.

### 비교

| 항목 | 기본 Tool Calling 예제 | ReAct 에이전트 |
|---|---|---|
| 도구 선택 | LLM | LLM |
| 도구 실행 | 개발자 코드가 연결 | `ToolNode` 등으로 자동화 가능 |
| 반복 처리 | 개발자가 루프 구현 | 그래프에서 루프 구성 |
| 종료 결정 | 개발자 로직 구성 | 도구 호출 여부 + 종료 조건 |
| 핵심 개념 | 외부 기능 사용 | 판단·실행·재판단의 반복 패턴 |

> **정리:** Tool Calling은 기능이고, ReAct는 그 기능을 반복적으로 이용하는 에이전트 설계 패턴이다.

## 4. ReAct 실행 예시

`"3과 5를 곱하고 10을 더해줘"` 요청을 받으면 다음처럼 진행할 수 있다.

`Agent 판단 → multiply(3,5) → 결과 15 → Agent 재판단 → add(15,10) → 결과 25 → 최종 응답`

- 실행 순서와 호출 횟수는 모델에 따라 달라질 수 있다.
- ReAct의 판단은 비공개 내부 사고 과정을 그대로 보여준다는 뜻이 아니다.

## 5. `create_react_agent()`로 간단히 구성

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def multiply(a: int, b: int) -> int:
    """두 정수를 곱합니다."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """두 정수를 더합니다."""
    return a + b

model = ChatOpenAI(model="gpt-5-mini")
agent = create_react_agent(model, [multiply, add])

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "3과 5를 곱한 뒤 10을 더해줘"}
    ]
})
print(result["messages"][-1].content)
```

- `create_react_agent()`는 **원문에서 사용하는 LangGraph 사전 구축 방식**이다.
- 최신 LangChain 환경에서는 `langchain.agents.create_agent()`도 고려한다.
- 버전에 따라 권장 API와 세부 옵션이 달라질 수 있다.
- 사전 구축 에이전트는 `agent → tools → agent` 흐름을 내부적으로 제공한다.

## 6. 내부 구성: Node, State, Edge

| 구성 요소 | 역할 |
|---|---|
| `State` | 질문, LLM 응답, 도구 결과 등 공통 상태 저장 |
| `agent` 노드 | 현재 메시지로 LLM 호출 및 다음 행동 판단 |
| `tools` 노드 | 요청된 도구 실행 |
| 조건부 Edge | 도구 호출 여부에 따라 tools 또는 END 선택 |
| `tools → agent` Edge | 도구 실행 후 LLM으로 되돌아감 |

```text
START → agent ──(tool_calls 있음)──→ tools
            ↑                         │
            └─────────────────────────┘
          └──(tool_calls 없음)──→ END
```

## 7. `StateGraph`로 ReAct 직접 구현

```python
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def multiply(a: int, b: int) -> int:
    """두 정수를 곱합니다."""
    return a * b

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatOpenAI(model="gpt-5-mini")
model_with_tools = model.bind_tools([multiply])

def agent_node(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode([multiply]))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
app = graph.compile()

result = app.invoke({
    "messages": [{"role": "user", "content": "5와 4를 곱해줘"}]
})
print(result["messages"][-1].content)
```

### 7-1. `AgentState`와 `add_messages`

- `messages`는 사용자·LLM·도구 메시지를 담는 상태 필드다.
- `add_messages`는 새로운 메시지를 기존 메시지에 병합하는 reducer다.
- Tool 실행 결과가 누적되어 다음 LLM 호출에서 참고된다.
- 상태가 없으면 각 단계 사이에 맥락을 전달하기 어렵다.

### 7-2. `agent_node`

- 현재 `state["messages"]`를 모델에 입력한다.
- 모델은 `AIMessage`를 반환한다.
- `AIMessage`에는 최종 텍스트 또는 `tool_calls`가 포함될 수 있다.
- `return {"messages": [response]}`로 새 메시지를 상태에 반영한다.

### 7-3. `ToolNode`

- 모델이 생성한 도구 이름과 인자를 확인한다.
- 해당 Python 함수 또는 등록된 도구를 실제 실행한다.
- 결과를 `ToolMessage`로 반환해 상태에 반영한다.
- **`bind_tools()`만 했을 때와 달리 실제 실행 과정이 연결된다.**

### 7-4. `tools_condition`

- 마지막 LLM 응답에 `tool_calls`가 있는지 확인한다.
- 있으면 `tools` 노드로 분기한다.
- 없으면 그래프의 `END`로 진행한다.
- 분기는 LLM의 출력과 그래프의 종료 규칙에 따라 결정된다.

### 7-5. `graph.add_edge("tools", "agent")`

- 도구 결과를 확인한 후 다시 LLM을 호출한다.
- 여러 도구를 순차적으로 사용할 수 있는 반복 구조의 핵심이다.
- 반복 횟수·시간·비용 제한이 없다면 불필요한 호출이 길어질 수 있다.

## 8. 사전 구축과 직접 구현 비교

- 사전 구축: 코드가 적고 기본 반복 구조를 자동 제공하므로 빠른 실습에 적합하다.
- 직접 구현: 노드, 분기, 커스텀 상태 등을 제어할 수 있어 복잡한 서비스에 적합하다.

## 9. 스트리밍과 실행 확인

```python
result = app.invoke(input_data)  # 완료 후 최종 상태 반환
for chunk in app.stream(input_data, stream_mode="updates"):
    print(chunk)  # 노드별 변경 사항 출력
```

| 모드 | 반환 정보 | 용도 |
|---|---|---|
| `values` | 단계별 전체 상태 | 상태 변화 확인 |
| `updates` | 노드별 변경 | 진행 로그 |
| `messages` | LLM 메시지·토큰 | 실시간 채팅 UI |

- 스트리밍으로 보이는 것은 실행 이벤트이지 모델의 내부 사고 과정 전체가 아니다.

## 10. 오류 처리와 반복 제한

```python
# ToolNode 오류 처리
ToolNode([multiply], handle_tool_errors=True)

# 실행 단계 제한 (도구 호출 10회 제한이 아님)
result = app.invoke(input_data, config={"recursion_limit": 10})
```

- 도구 오류를 메시지로 반환하면 Agent가 다른 방법을 시도할 수 있다.
- 모든 오류가 자동 복구되는 것은 아니므로 타임아웃·재시도 한도를 정한다.
- `recursion_limit`은 **그래프 실행 단계 수**를 제한한다.
- 제한 도달 시 `GraphRecursionError`를 처리하거나 사용자에게 대안을 안내한다.

## 11. RAG와 결합: Agentic RAG

- Ch.07의 Retriever 또는 Ch.08의 RAG 검색 기능을 Tool로 제공할 수 있다.
- Agent가 **검색할지 여부와 추가 검색 필요성**을 결정하도록 설계할 수 있다.

```text
질문 → ReAct Agent
             ↓
        RAG 검색 Tool
             ↓
        검색 결과 확인
             ↓
        정보가 충분한가?
        ├─ 아니오 → 검색어 수정 → 재검색
        └─ 예 → 근거를 바탕으로 답변
```

| 비교 | 일반 RAG | Agentic RAG |
|---|---|---|
| 검색 흐름 | 미리 정해진 파이프라인 | 필요 시 유동적 검색 |
| 검색 횟수 | 통상 사전에 지정 | 여러 차례 실행 가능 |
| 장점 | 단순함·낮은 비용 | 복합 질문 대응 유연성 |
| 단점 | 복잡한 질문에 약할 수 있음 | 호출 비용과 지연 증가 |

- 항상 Agentic RAG가 우수한 것은 아니다.
- 단순한 공식 문서 Q&A는 일반 RAG가 더 안정적일 수 있다.

## 12. 자주 헷갈리는 핵심 질문

**Q. `bind_tools()`가 함수를 실행하나?**  
A. 아니다. 도구의 명세를 모델에 제공할 뿐 실제 실행기는 별도로 필요하다.

**Q. `ToolNode`도 어떤 도구를 쓸지 판단하나?**  
A. 일반적으로 선택은 LLM, 실제 실행은 `ToolNode`가 담당한다.

**Q. ReAct는 무조건 여러 도구를 호출하나?**  
A. 아니다. 도구 없이 답하거나 필요한 만큼 반복할 수 있다.

**Q. ReAct와 멀티 에이전트는 같은가?**  
A. 아니다. ReAct는 하나의 에이전트 내부 실행 패턴이고, 멀티 에이전트는 여러 에이전트의 협업 구조다.

