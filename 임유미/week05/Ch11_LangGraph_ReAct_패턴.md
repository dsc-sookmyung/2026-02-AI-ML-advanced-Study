# LangGraph 에이전트: ReAct 패턴 구현 (Ch.11)

<aside>

- ReAct의 추론-행동 반복 원리
- `create_react_agent`로 사전 구축 에이전트 만들기
- `StateGraph`로 커스텀 ReAct 에이전트 직접 구축
- 스트리밍으로 중간 단계 관찰
- 오류 처리와 재시도 전략
</aside>

## ReAct 패턴

> **Reasoning(추론) + Acting(행동)의 반복 루프**
> 
> LLM이 한 번에 답하지 않고 **생각 → 행동 → 관찰**을 반복하며 답에 접근. 종료 조건은 **LLM이 tool_calls 없이 최종 답변을 생성할 때**

| 단계 | 역할 | LangGraph 대응 |
| --- | --- | --- |
| Thought(추론) | 어떤 도구를 쓸지 판단 | agent 노드 (LLM + `bind_tools`) |
| Action(행동) | 선택한 도구 실행 | tools 노드 (`ToolNode`) |
| Observation(관찰) | 실행 결과 확인 | `ToolMessage`가 상태에 추가 |
| Final Answer | 정보가 충분하면 답변 | 조건부 엣지가 `END`로 분기 |

**단순 도구 호출(Ch.5) vs ReAct**

| 구분 | 단순 도구 호출 | ReAct |
| --- | --- | --- |
| 도구 호출 횟수 | 1회 | 여러 회 반복 |
| 그래프 구조 | 직선 (A → B → C) | 순환 (A → B → A → B …) |
| 종료 조건 | 도구 결과 반환 | LLM이 최종 답변 결정 |
| 구현 | LCEL 체인 | LangGraph StateGraph |

**내부 동작 흐름**

1. 사용자 질문 수신
2. agent 노드: LLM이 메시지 히스토리와 도구 목록을 보고 판단
3. 조건 분기: `tool_calls` 있으면 tools로, 없으면 END로
4. tools 노드: 도구 실행 후 `ToolMessage`로 변환
5. 결과를 히스토리에 추가하고 2번으로 복귀
6. 충분하면 텍스트 답변 생성

→ 개발자가 순서를 하드코딩하지 않고, **LLM이 언제 어떤 도구를 쓸지, 언제 멈출지 스스로 판단**

## create_react_agent (사전 구축)

> 모델과 도구만 넘기면 ReAct 그래프를 한 줄로 생성 (밀키트)

내부적으로 자동 수행하는 일: agent 노드 생성 → `ToolNode` 생성 → `tools_condition`으로 조건부 엣지 → `tools → agent` 순환 엣지

```python
agent = create_react_agent(model, [multiply, add])
result = agent.invoke({"messages": [{"role": "user", "content": "23 곱하기 17은?"}]})
```

| 파라미터 | 설명 | 필수 |
| --- | --- | --- |
| `model` | LLM (문자열 또는 인스턴스) | O |
| `tools` | 도구 리스트 | O |
| `prompt` | 시스템 프롬프트 | X |
| `checkpointer` | 상태 저장 (Ch.12) | X |
| `state_schema` | 커스텀 상태 스키마 | X |
| `response_format` | 응답 구조화 스키마 | X |

**구성 요소**

| 구성 요소 | 역할 |
| --- | --- |
| `ToolNode` | `tool_calls`를 읽어 도구 실행 → `ToolMessage` 변환, 여러 도구는 **병렬 실행**, `handle_tool_errors`로 예외 처리 |
| `tools_condition` | 마지막 메시지에 `tool_calls`가 있으면 `"tools"`, 없으면 `END` 반환 |

**주의: 도구 description이 성능을 결정**

- docstring은 LLM에게 주는 "사용 설명서"
- 포함할 것: ① 무엇을 하는지 ② 언제 쓰는지 ③ 입출력 예시
- `"데이터를 처리합니다"`, `"계산합니다"` 같은 모호한 설명은 잘못된 도구 선택의 원인

## 커스텀 ReAct 에이전트 (StateGraph)

> 로깅, 특수 분기, 반복 횟수 제한 등 **세밀한 제어**가 필요할 때 직접 구축

| 구성 요소 | 구현 |
| --- | --- |
| 상태 | `TypedDict` + `Annotated[list, add_messages]` |
| agent 노드 | `bind_tools()`된 LLM 호출 |
| tools 노드 | `ToolNode` 또는 커스텀 함수 |
| 라우터 | `tool_calls` 유무로 `"tools"` / `END` 반환 |

```python
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode([multiply]))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")   # 순환의 핵심
app = graph.compile()
```

| 상황 | 권장 방식 |
| --- | --- |
| 빠른 프로토타이핑 / 표준 ReAct면 충분 | `create_react_agent()` |
| 반복 횟수 제한 | 커스텀 그래프 |
| 도구 호출 전/후 커스텀 로직 | 커스텀 그래프 |
| Human-in-the-Loop 세밀 제어 | 커스텀 그래프 |

**반복 횟수 제한**: 상태에 `iteration_count`를 추가하고, agent 노드에서 +1, 라우터에서 `MAX_ITERATIONS`(보통 5~10) 이상이면 `"end"` 반환. `create_react_agent`는 `config={"recursion_limit": 10}`으로 제한

## 스트리밍과 중간 단계 관찰

| stream_mode | 출력 | 용도 |
| --- | --- | --- |
| `"values"` | 각 단계의 전체 상태 | 디버깅 |
| `"updates"` | 해당 단계의 변경분만 | 프로덕션 로깅, 진행 상황 표시 |
| `"messages"` | LLM 토큰 단위 | 챗봇 타이핑 효과 |

- 복수 모드 동시 사용 가능: `stream_mode=["values", "messages"]`
- `astream_events(version="v2")`: 가장 세밀한 비동기 이벤트 추적

| 이벤트 | 의미 |
| --- | --- |
| `on_chat_model_stream` | LLM 토큰 생성 |
| `on_tool_start` / `on_tool_end` | 도구 실행 시작 / 완료 |
| `on_chain_start` / `on_chain_end` | 체인·그래프 시작 / 완료 |

## 오류 처리와 재시도

| 전략 | 구현 | 단점 |
| --- | --- | --- |
| ToolNode 기본 처리 | `handle_tool_errors=True` | 재시도 불가 |
| try-except 래핑 | 도구 내부에서 처리 | 반복 코드 |
| 폴백 도구 | 대체 도구 설정 | 설계 복잡 |
| 에러 메시지 주입 | 에러를 `ToolMessage`로 LLM에 전달 | LLM 비용 증가 |

**3중 안전 장치**

1. `recursion_limit`: 무한 루프 방지 (기본값 25)
2. 에러 메시지 주입: LLM이 스스로 수정
3. 폴백 응답: `GraphRecursionError`를 잡아 대체 안내 제공

- 에러 메시지에는 **① 무엇이 잘못됐는지 ② 어떻게 대응할지**를 포함. `"에러 발생"`처럼 모호하면 LLM이 같은 실수를 반복
- `recursion_limit`은 **노드 실행 단계 수** 기준. agent → tools가 1회 반복이면 2단계 소비, `recursion_limit=10`이면 약 5회 반복

### 정리

- ReAct = **추론 + 행동의 반복 루프**, LLM이 도구 호출 여부를 자율 판단
- Ch.10 StateGraph + Ch.5 Tool Calling을 결합한 구조
- 프로토타입은 `create_react_agent`, 커스텀 로직이 필요하면 `StateGraph`
- 도구 description이 에이전트 성능을 좌우
- 스트리밍은 `values`(디버깅), `updates`(로깅), `messages`(UI)
- 안정성은 **recursion_limit + 에러 주입 + 폴백**