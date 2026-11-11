# Week 5 (ReAct Pattern, Multi-Agent System)

### 📒 강의 개요

11장 - ReAct Pattern

[https://wikidocs.net/333795](https://wikidocs.net/333795)

13장 - Multi-Agent System
[https://wikidocs.net/333797](https://wikidocs.net/333797)


## Cp. 11, 13 - **LangGraph 에이전트: ReAct 패턴 구현** / **멀티 에이전트 시스템: Supervisor와 Swarm**

### 핵심 개념 및 용어 정리

---

#### Cp.11 LangGraph 에이전트: ReAct 패턴 구현

- ReAct 패턴 - 추론(Reason) + 행동(Act)의 반복
    - LLM은 혼자서 실시간 검색이나 계산을 할 수는 없지만, 어떤 도구를 써야할지 판단하는 능력을 뛰어남
        - 따라서 LLM의 추론 능력 + 외부 도구의 실행 능력 ⇒ 복잡한 문제를 단계적으로 해결하는게 ReAct 패턴
    - ReAct = **Re**asoning(추론) + **Act**ing(행동)
        - 핵심 아이디어 : LLM이 한 번에 최종 답을 내리는 대신, “생각 → 행동 → 관찰”을 반복하여 점진적으로 답에 접근
        - 반복루프 종료 조건 : LLM이 더 이상 도구를 호출하지 않고 최종 답변 직접 생성할 때
        1. 추론(Thought) -  어떤 도구 써야 할지 판단
        2. 행동(Action) - 선택한 도구를 실제로 실행
        3. 관찰(Observation) - 도구 실행 결과 확인
        4. 최종 답변(Final Answer) - 충분한 정보가 모이면 답변 생성
    - ReAct vs 단순 도구 호출 (Tool Calling)
        - Tool Calling은 LLM이 1번 도구를 호출하고 끝나는 구조
        - ReAct 패턴은 이것을 순환 구조로 확장함
    - ReAct 패턴 내부 동작 흐름
        - 사용자 질문 수신
        - LLM 추론 (agent 노드) - 도구를 호출할지 직접 답변할지 판단
        - 조건 분기 - LLM 응답에 tool_calls → 도구 실행 노드 / 없으면 종료(END)
        - 도구 실행 (tools 노드) - 선택된 도구 실제 실행, 결과 → ToolMessage
        - 결과 다시 LLM에 전달 - ToolMessage가 메시지 히스토리에 추가, 다시 2번으로
        - 최종 답변 - LLM이 정보 충분하다고 판단 → 도구 호출하지 않고 텍스트 답변 생성
        
        <aside>
        💡
        
        ReAct 에이전트의 지능 ⇒ LLM이 언제 어떤 도구 호출할지, 언제 멈출지 스스로 판단 (개발자가 순서 하드코딩 X
        
        </aside>
        
    - 실제 ReAct 에이전트에서는 위 과정이 그래프(Graph) 구조로 자동 수행됨
- create_react_agent
    - 사전 구축된 ReAct 에이전트
    - 매번 StateGraph를 정의하고 노드와 엣지를 연결하고 조건부 분기 설정하는 것은 반복적인 작업
    ⇒ create_react_agent()라는 사전 구축(prebuilt) 함수 제공
    ⇒ 모델과 도구 목록만 전달 ⇒ 완성
    - create_react_agent()는 langgraph.prebuilt 모듈에서 제공하는 팩토리 함수
        - agent 노드 생성 - LLM에 도구 바인딩(bind_tools), 메시지 처리 노드
        - tools 노드 생성 - ToolNode로 도구 실행 노드 만듦
        - 조건부 edge 연결 - tools_condition으로 도구 호출 여부에 따른 분기 설정
        - 순환 엣지 연결 - tools → agent로 돌아가는 edge 추가
- ToolNode
    - LangGraph 제공하는 사전 구축 도구 실행 노드
    - AIMessage의 tool_calls 필드 → 해당하는 도구 실행 → 결과를 ToolMessage로 변환하여 상태에 추가
- tools_condition
    - LangGraph가 제공하는 사전 구축된 조건 판별 함수
    - 상태의 마지막 메시지 검사
        - tool_calls 있으면 tools, 없으면 END 반환
    - create_react_agent() 내부에서 자동으로 조건부 엣지에 연결
    - 커스텀 에이전트에서 직접 사용할수도.
- prompt 파라미터 ⇒ 시스템 프롬프트 전달하면 에이전트 행동 방식 제어할 수 있음 (역할, 언어, 응답 스타일 제어)
    - 실수 조심) 도구의 description을 대충 모호하게 작성하면 LLM이 잘못된 도구를 선택할수도
- 커스텀 ReAct 에이전트
    - 실무에서는 기본 ReAct 구조만으로 부족한 경우 O
    - ⇒ StateGRaph로 직접 그래프를 구축해야함
    - 커스텀 ReAct 에이전트 4가지 요소
        - 상태(state) - 메시지 리스트 관리
        - agent 노드 - LLM 추론 수행
        - tools 노드 - 도구 실행
        - 라우터 함수 - 분기 결정
- 에이전트 스트리밍과 중간 단계 관찰
    - ReAct 에이전트 → 추론-행동을 여러 번 반복하기 때문에, 최종 답변이 나올 때까지 시간이 걸릴 수 있음.
    - 따라서 streaming을 사용하여 에이전트가 생각하는 과정을 실시간으로 제시할 수 있음
    - LangGraph의 .stream() 메서드는 stream_mode 파라미터로 출력 헝식을 제어함
    - astream_events로 비동기 환경에서 에이전트 내부의 이벤트들을 세밀하게 추적할 수 있음
- 에이전트의 오류 처리와 재시도 전략
    - 실무에서 에이전트는 다양한 이유로 실패할 수 있기 때문에, 이러한 오류를 적절히 처리해야함
    - 따라서 robust한 에이전트를 만드려면 오류 처리 전략이 필수적임.
    - 가장 효과적인 전략 - 에러 메시지를 LLM에게 다시 전달하기
        - LLM은 에러 메시지를 보고 인자 수정하거나 다른 도구 선택할 수 있음
    - 에러 처리에서 에러를 잡고도 의미 있는 메시지를 전달하지 않을 수 있음

#### Cp.13 **멀티 에이전트 시스템: Supervisor와 Swarm**

- 멀티 에이전트 아키텍처 (패턴)
    - ⇒ 여러 개의 전문화된 AI 에이전트가 서로 협력하여 복잡한 작업 수행하는 시스템 구조
    - 각 에이전트는 고유한 역할, 도구, 프롬프트 O / 정해진 규칙에 따라 작업 주고받음
    - LangGraph 제공 대표적인  멀티 에이전트 패턴
        - Supervisor - 중앙 지휘자가 에이전트 선택하고 작업을 위임
        - Swarm(스웜) - 에이전트들이 핸드오프 도구로 자율적으로 작업 넘김
        - Hierarchical (계층적) - Supervisor들의 Superisor, 팀 단위 계층 구조
    - 패턴 선택 기준
        - 핵심원칙 : 단순한 작업 → Supervisor로 시작, 에이전트 간 직접 대화가 필요하면 Swarm, 규모가 커지면 Hierarchical로 확장
- Supervisor 패턴 - 중앙 지휘자가 에이전트 조율
    - (팀장 역할) 중앙 지휘 에이전트 Supervisor가 사용자 요청 분석하고, 적절한 전문 에이전트에게 작업을 위임하며 최종결과 취합
    - create_supervisor() 함수 사용
        - Supervisor가 사용자 요청을 받아서 → 적절한 에이전트 선택 → 선택된 에이전트 작업 수행 → 결과 Supervisor에게 돌아옴 → 모든 작업이 완료되면 Supervisor가 최종 답변 생성
    - 내부적으로는 Supervisor도 하나의 LLM 에이전트임. Tool Calling 메커니즘을 활용한 에이전트 전환 사용.
    - supervisor 행동을 더 세밀하게 제어 → 커스텀 프롬프트 활용
    - 주의) agent의 name 파라미터를 빠뜨리면 Supervisor가 에이전트 구분 못함
    - Supervisor 패턴 내부 동작 정리 (Supervisor → 에이전트 → supervisor → 에이전트 → … → 최종 답변)
        1. 사용자 요청 수신 - messages에 HumanMessage 추가
        2. Supervisor LLM 호출 - 프롬프트, 핸드오프 도구 목록 전달
        3. agent 선택 - tool_calls에 transfer_to_{name} 반환
        4. agent 실행 - 선택된 agent node로 제어권 이동
        5. 결과 반환 - agent의 AIMessage가 messages에 추가
        6. Supervisor 재호출 - 다음 agent 선택 도는 최종 답변 생성
- Swarm 패턴 : dpdlwjsxm rks wkdbf gosemdhvm
    - 굳이 Supervisor가 중간에서 판단할 필요가 없는 경우라면? 매번 Supervisor가 LLM을 호출한다면 비용과 시간이 낭비될 수 있지 않을까?
    - Swarm 패턴 ⇒ 중앙 지휘자 없이, 에이전트들이 핸드오프 도구를 사용하여 자율적으로 다음 에이전트에게 작업을 넘기는 탈중앙화 협업 패턴
        - ex) 릴레이 경주 - 각 주자(agnet)가 자신의 구간(전문 분야) 달리기
        - LLM 호출 오버헤드는 낮지만, 전체 흐름 가시성도 확보가 어려움 (개별 에이전트 시점에서만 판단하기 때문에)
    - Swarm은 create_swarm() 함수로 구현하며, 핵심은 create_handoff_tool()함수
        - 각 에이전트에게 핸드오프 도구 등록 → 에이전트가 스스로 판단 → 제어권이 직접 이동 → 대화 맥락 유지 (메시지 히스토리 전달)
        - 핸드오프 도구 - 에이전트가 ‘이 대화를 다른 에이전트에게 넘긴다’고 선언하는 특수 도구
        - 기본 핸드오프 도구는 모든 메시지를 그대로 전달하기 때문에, 커스텀을 통해 요약하거나 추가 메타데이터 붙일 수 있음
    - 주의) 핸드오프 경로를 순환없이 설계하는 것, 체크포인터 없이 멀티 턴 대화 시도하는 것
        - 체크포인터 없이 compile하면, 각 invoke 호출마다 상태 초기화되어 active_agent가 항상 default로 리셋된다고 함. 핸드오프로 agent_b가 활성화되어도, 다음 호출에서 다시 agnet_a부터 시작
    - Swarm 공유 상태 → active_agent라는 특별한 필드
        - 현재 어떤 에이전트가 대화 담당하는지 추적
        - 멀티턴 대화에서 마지막으로 활성화된 에이전트가 자동으로 이어받도록 함.
        - 체크포인터가 없으면 이 상태가 유지되지 않음 (다라서 필수)
- 계층적 멀티 에이전트 - Supervisor of Supervisors
    - 계층적 패턴 ⇒ 여러 개의 Supervisor 팀을 상위 Supervisor가 조율하는 다단계 구조
        - 각 팀은 독립된 서브그래프로 구성, 상위 Supervisor가 팀 단위로 작업 위임함
    - 장점
        - 관심사 분리, 확장성, 독립적 테스트
    - 계측 구조 설계 시 고려 사항
        - 팀당 에이전트 수 : 2~4개 권장 (Supervisor의 선택 정확도 유지)
        - 계층 깊이 : 2~3단계 권장 (깊이 깊으면 latency 증가)
        - 팀 간 의존성 : 최소화 (독립적 테스트와 유지보수 용이)
        - output_mode : 하위 팀 - last_message, 최상위 - full_history (중간 과정 노이즈 제거)
    - 처음부터 계층구조를 만들기 보다는, 먼저 단일 Supervisor로 시작하기
- 에이전트간 상태 공유와 통신
    - 멀티 에이전트에서 가장 중요한 부분 - 에이전트간 상태 공유
        - 리서치 에이전트가 수집항 자료 → 작성 에이턴트가 참조 → 그 결과를 검수 에이전트로 전달
        - 상태공유가 제대로 되지 않으면 이전 에이전트의 대화 맥락이 사라져서, 처음부터 다시 시작
    - 메시지 패싱(Passing) 방식
        - 에이전트들이 공유 상태의 메시지 리스트에 메시지 추가하고, 다른 에이전트가 이 메시지 읽어 맥락 파악하는 방식 (like 게시판)
        - create_supervisor()와 create_swarm()이 내부적으로 MessageState 공유상태로 사용
    - full_history vs last_message
        - Supervisor의 output_mode 파라미터가 에이전트 간 메시지 전달 방식 제어 (→ 비용과 품질의 균형)
        - full_history : 모든 에이전트 전체 대화 내역 유지 (팀 내부에서 추천)
        - last_message : 각 에이전트의 마지막 메시지만 유지 (팀 간에 추천)