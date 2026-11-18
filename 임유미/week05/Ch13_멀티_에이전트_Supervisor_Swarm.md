# 멀티 에이전트 시스템: Supervisor와 Swarm (Ch.13)

<aside>

- 멀티 에이전트 아키텍처 3가지 패턴 비교
- Supervisor 패턴 (중앙 지휘형)
- Swarm 패턴 (자율 협업형)
- 계층적 멀티 에이전트 (Supervisor of Supervisors)
- 에이전트 간 상태 공유와 `output_mode`
</aside>

## 멀티 에이전트가 필요한 이유

> **멀티 에이전트 = 역할·도구·프롬프트가 다른 전문 에이전트 여러 개가 협력하는 구조**

- 하나의 에이전트에 모든 역할을 주면 프롬프트가 길어지고 전문성이 떨어짐
- 도구가 10개 이상이거나 역할이 서로 다르면 분리가 효율적 (셰프·서버·캐셔가 나뉜 식당)

| 패턴 | 핵심 | 비유 | API |
| --- | --- | --- | --- |
| Supervisor | 중앙 지휘자가 위임 | 오케스트라 지휘자 | `create_supervisor()` |
| Swarm | 핸드오프 도구로 자율 전달 | 릴레이 경주 | `create_swarm()` |
| Hierarchical | Supervisor들의 Supervisor | 대기업 조직도 | Supervisor 중첩 |

| 기준 | Supervisor | Swarm | Hierarchical |
| --- | --- | --- | --- |
| 작업 흐름 | 중앙에서 순서 결정 | 에이전트가 자율 결정 | 팀 단위 분리 |
| 적합한 상황 | 작업 순서가 중요 | 에이전트 간 직접 전환 빈번 | 팀이 여러 개, 대규모 |
| 복잡도 | 낮음 | 중간 | 높음 |
| LLM 호출 수 | 중간 | 적음 | 많음 |
| 대표 사례 | 리서치 → 작성 → 검수 | 고객 지원 라우팅 | 대규모 개발 팀 |

**선택 원칙**: Supervisor로 시작 → 직접 대화가 필요하면 Swarm → 규모가 커지면 Hierarchical

## Supervisor 패턴

> 중앙 Supervisor(LLM)가 요청을 분석해 전문 에이전트에 위임하고 결과를 취합

**동작 원리**

1. Supervisor가 요청 수신
2. 등록된 에이전트 중 적합한 에이전트 선택 (`transfer_to_{name}` **핸드오프 도구 호출**)
3. 선택된 에이전트가 작업 수행
4. 결과가 Supervisor로 복귀
5. 다음 에이전트 선택 또는 최종 답변 생성

→ Supervisor도 하나의 LLM 에이전트이고, 에이전트 전환은 **도구 호출 메커니즘**으로 이뤄짐

```python
researcher = create_agent(model=model, tools=[search_web], name="researcher", system_prompt="...")
writer = create_agent(model=model, tools=[write_document], name="writer", system_prompt="...")

workflow = create_supervisor(agents=[researcher, writer], model=model, prompt="...")
app = workflow.compile()
```

| 파라미터 | 설명 |
| --- | --- |
| `agents` | 관리할 에이전트 리스트 |
| `model` | Supervisor의 LLM |
| `prompt` | 작업 순서·검증 기준·출력 형식 등 행동 지침 |
| `output_mode` | `"full_history"` / `"last_message"` |
| `supervisor_name` | Supervisor 노드 이름 (기본 `"supervisor"`) |
| `tools` | Supervisor 자체에 추가할 도구 (선택) |

**주의**

- `create_agent`의 **`name`은 필수**. 없으면 핸드오프 도구가 제대로 생성되지 않음
- 비용: 에이전트 3개를 1번씩 호출하면 Supervisor 4회 + 에이전트 3회 = **최소 7회 LLM 호출**
- 순서 강제는 프롬프트 기반이라 100% 보장이 아님. 엄격해야 하면 `StateGraph`에서 엣지로 직접 정의

## Swarm 패턴

> 중앙 지휘자 없이 에이전트들이 **핸드오프 도구**로 다음 에이전트에게 직접 작업을 넘기는 탈중앙화 구조

| 비교 | Supervisor | Swarm |
| --- | --- | --- |
| 제어 | 중앙 지휘 | 탈중앙화 |
| 전환 | Supervisor 경유 | 에이전트 간 직접 |
| LLM 호출 | 높음 | 낮음 |
| 흐름 가시성 | 높음 | 낮음 (디버깅 까다로움) |

**동작 원리**

1. 각 에이전트에 `create_handoff_tool(agent_name=...)` 등록
2. 에이전트가 "내 전문 분야가 아니다"라고 판단하면 핸드오프 호출
3. 공유 상태의 **`active_agent`** 갱신 → 다음 에이전트가 바로 실행
4. 메시지 히스토리가 그대로 전달되어 맥락 유지

핸드오프 도구가 하는 일: ① `ToolMessage`로 전환 확인 ② `Command(goto=...)`로 다음 노드 지정 ③ `active_agent` 업데이트

```python
transfer_to_billing = create_handoff_tool(agent_name="billing_agent", description="환불·결제 문의일 때 전환")
workflow = create_swarm(agents=[general_agent, billing_agent, tech_agent], default_active_agent="general_agent")
app = workflow.compile(checkpointer=InMemorySaver())   # 체크포인터 필수
```

- `create_handoff_tool` 옵션: `name`(도구 이름 커스텀), `description`(전환 시점 판단 기준)
- **체크포인터 필수**: 없으면 `invoke`마다 `active_agent`가 default로 리셋. 같은 `thread_id`를 써야 상태 유지
- 에이전트가 가진 핸드오프 도구 목록 = **허용된 전환 경로**. A에게 `transfer_to_B`만 주면 A → C는 불가
- 대표 패턴: 1차 상담 → 2차 전문가 에스컬레이션 (양방향 핸드오프)

## 계층적 패턴 (Supervisor of Supervisors)

> 여러 Supervisor 팀을 상위 Supervisor가 조율. 각 팀은 독립 서브그래프

- 장점: **관심사 분리 / 확장성 / 독립 테스트**
- 하위 팀은 `.compile(name="research_team")`으로 이름을 지정해 상위 Supervisor의 `agents`에 등록

| 고려사항 | 권장 | 이유 |
| --- | --- | --- |
| 팀당 에이전트 수 | 2~4개 | 선택 정확도 유지 |
| 계층 깊이 | 2~3단계 | 깊을수록 레이턴시 증가 |
| 팀 간 의존성 | 최소화 | 독립 테스트·유지보수 |
| output_mode | 하위 팀 `last_message`, 최상위 `full_history` | 중간 과정 노이즈 제거 |

- 처음부터 계층으로 만들지 말고, **단일 Supervisor로 시작해 에이전트가 5개 이상이면 분리**

## 상태 공유와 통신

> 에이전트 간 통신 = **메시지 패싱**. 공유 상태(`MessagesState`)의 메시지 리스트에 쓰고 읽음 (게시판)

| output_mode | 동작 | 적합한 상황 |
| --- | --- | --- |
| `"full_history"` | 모든 에이전트의 전체 대화 유지 | 맥락 공유가 중요할 때 (토큰·비용 증가) |
| `"last_message"` | 각 에이전트의 마지막 메시지만 유지 | 결과만 필요할 때 (토큰 절약) |

- 상태 공유가 안 되면 다음 에이전트가 "자료가 없다"고 답하거나 처음부터 다시 시작
- 프로덕션에서는 `InMemorySaver` 대신 `SqliteSaver` / `PostgresSaver`
- 디버깅은 **LangSmith 트레이싱** (`LANGCHAIN_TRACING_V2=true`)

## 비용·운영 팁

- 비용 ≈ 에이전트 수 × LLM 호출 수
- Supervisor는 가벼운 모델, 전문 에이전트만 고성능 모델을 쓰면 효율적
- Swarm은 중간 Supervisor 호출이 없어 더 저렴
- 무한 루프는 `recursion_limit`(기본 25)으로 제한

### 정리

- 멀티 에이전트 = **전문 에이전트 + 조율 방식**. 패턴은 Supervisor / Swarm / Hierarchical
- Supervisor는 **순서 제어**, Swarm은 **빠른 직접 전환**, Hierarchical은 **대규모 팀 관리**
- 에이전트 전환은 모두 **핸드오프 도구(도구 호출)** 로 구현
- `name` 지정(Supervisor), 체크포인터(Swarm)는 빠뜨리기 쉬운 필수 조건
- 통신은 메시지 패싱, `output_mode`로 전달 범위 조절
- 처음엔 단순하게 시작하고, 필요할 때만 확장