# 위키독스 01-01 & 05

## 01-01 AI Agent란?

- Chatbot: LLM을 통해 질문에 답변하는 시스템
- RAG: 외부 데이터를 검색하여 LLM의 답변에 필요한 문맥을 제공하는 방식
- AI Agent: LLM이 상황에 따라 도구를 선택하고 결과를 확인하며 작업을 수행하는 시스템
- Agent = LLM + Tools + Loop
  - LLM: 의사결정
  - Tools: 실제 작업 수행
  - Loop: 결과 확인 및 반복
- ReAct: Thought → Action → Observation을 반복하고, 충분한 정보가 모이면 Answer 생성
- Hybrid Architecture: 판단과 자연어 처리는 LLM, 정확한 계산과 실행은 코드·규칙·도구가 담당
- LLM은 도구 호출을 요청하며 실제 도구 실행은 외부 프로그램이 담당한다.

## Ch.05 **Tool Calling과 Function Calling**

### 1. Tool Calling 기초

- `Tool Calling`: LLM이 필요한 도구와 호출 인자를 선택하고, 구조화된 요청을 생성하는 기능
- `Function Calling`: Tool Calling의 이전 명칭
- `@tool`: Python 함수를 LangChain Tool 객체로 변환하는 데코레이터
  - `Docstring`: 도구의 기능을 설명하여 LLM의 도구 선택을 도움
  - `Type Hint`: 도구의 인자 스키마를 정의하는 데 사용
- `bind_tools()`: 모델이 사용할 수 있는 도구의 정보를 연결
  - `tool_calls`: LLM이 생성한 도구 호출 요청이 담기는 필드

주의) 도구 바인딩 ≠ 실제 실행

→ LLM이 요청한 도구는 외부 프로그램이 실행해야 함

| 단계             | 담당 역할                            |
| ---------------- | ------------------------------------ |
| `@tool`          | 도구 정의                            |
| `bind_tools()`   | 도구 정보를 LLM에 연결               |
| `model.invoke()` | LLM 호출 및 도구 호출 요청 생성 가능 |
| `tool.invoke()`  | 외부 프로그램에서 실제 도구 실행     |

### 2. AIMessage

- LLM이 생성한 응답을 나타내는 메시지 객체
- 도구 호출이 필요한 경우 `tool_calls`에 도구 호출 요청이 담길 수 있음
- `tool_calls`의 주요 필드
  - `name` : 호출할 도구의 이름(str)
  - `args` : 도구에 전달할 인자 딕셔너리(dict)
  - `id` : 이 호출의 고유 식별자 → ToolMessage 매칭용(str)
  - `type` : 항상 “tool_call” (str)

⇒ `AIMessage`는 LLM이 생성한 응답 객체이며, `tool_calls`가 비어 있지 않다면 LLM이 외부 프로그램에 도구 실행을 요청했다는 의미!! 실제 도구가 실행된 것은 아니다

| 구분                   | 담당 역할                                                  |
| ---------------------- | ---------------------------------------------------------- |
| `AIMessage`            | LLM이 생성한 메시지 객체                                   |
| `AIMessage.content`    | LLM이 생성한 일반적인 메시지 내용                          |
| `AIMessage.tool_calls` | LLM이 생성한 도구 호출 요청 정보 (`name`, `args`, `id` 등) |
| `tool.invoke()`        | 외부 프로그램에서 실제 도구 실행                           |

⇒ 실제 도구 실행은 외부 프로그램이 담당하고, 실행 결과는 `ToolMessage`로 전달함을 기억

### 3. ToolMessage

- 외부 프로그램이 실행한 도구의 결과를 LLM에게 전달하는 메시지 타입
- `content`: 도구 실행 결과
- `tool_call_id`: `AIMessage.tool_calls`의 `id`와 일치해야 하며, 어떤 도구 호출에 대한 실행 결과인지 연결하는 식별자

### 4. Tool Execution Loop

`HumanMessage → AIMessage(tool_calls) → 도구 실행 → ToolMessage → AIMessage(최종 답변)`

- LLM은 도구 호출 요청을 생성 & 외부 프로그램이 실제 도구를 실행
- 도구 실행 결과를 메시지에 추가하여 LLM에 다시 전달
- 추가 도구 호출이 필요한 경우 위 과정을 반복

#### 추가로 헷갈리는 개념 정리

- `messages.append()`: 대화 기록에 메시지를 추가하며, LLM을 실행하지는 않는다.
- `ToolMessage`: 외부 도구의 실행 결과를 담는 메시지다. 최종 AI 답변과는 다르다.
- `model.invoke(messages)`: 저장된 대화 기록을 LLM에 전달해 새로운 응답을 생성한다. 도구 결과를 전달한 뒤에는 최종 답변 또는 추가 도구 호출 요청이 생성될 수 있다.
- 따라서 도구 실행 결과를 `messages.append()`로 추가한 뒤, LLM이 이를 활용해 최종 답변이나 다음 도구 호출을 생성하도록 `invoke()`를 다시 수행할 수 있다.

```python
from langchain_core.messages import HumanMessage, ToolMessage

# model과 multiply 도구는 이미 정의되어 있다고 가정
model_with_tools = model.bind_tools([multiply])

# 1. 사용자 질문
messages = [HumanMessage(content="3 곱하기 12는?")]

# 2. 첫 번째 LLM 호출
response = model_with_tools.invoke(messages)
messages.append(response)  # AIMessage(도구 호출 요청) 저장

# 3. LLM이 도구 호출을 요청했다면 실제 실행
if response.tool_calls:
    for tc in response.tool_calls:
        if tc["name"] == "multiply":
            result = multiply.invoke(tc["args"])
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"]
                )
            )

    # 4. 도구 결과를 포함해 LLM을 다시 호출
    final_response = model_with_tools.invoke(messages)
    messages.append(final_response)
    print(final_response.content)
else:
    print(response.content)

# 참고: 추가 tool_calls가 생성되면 다시 실행하는 루프가 필요함
```

- `multiply.invoke()`: 실제 도구를 실행하는 것
- `model_with_tools.invoke()`: 도구 정보가 연결된 LLM을 다시 호출해 새로운 `AIMessage`(최종 답변 또는 추가 도구 호출 요청)를 생성하는 것

**`invoke()` 메시지 관련 헷갈리는 거 정리**

| 코드                                 | 역할                                                            |
| ------------------------------------ | --------------------------------------------------------------- |
| `model.invoke(messages)`             | LLM을 호출하여 응답 생성                                        |
| `model_with_tools.invoke(messages)`  | 도구 정보가 연결된 LLM을 호출하여 응답 생성 또는 도구 호출 요청 |
| `multiply.invoke({"a": 3, "b": 12})` | 실제 `multiply` 도구 실행                                       |
| `messages.append(...)`               | 대화 기록에 메시지 추가 (실행 X)                                |

- `invoke()` 자체 ≠ LLM 전용 메서드
  - LangChain에서는 Tool이나 Chain 등을 실행할 때도 invoke()를 사용한다!
  - 결국 invoke()는 LangChain 객체를 실행하는 공통 메서드로 볼 수 있다!
  - `model.invoke()` = LLM 호출
  - `tool.invoke()` = 도구 실행

### 5. Multiple Tool Calls

- LLM은 하나의 응답에서 여러 도구 호출 요청을 생성할 수 있음
- `AIMessage.tool_calls`의 각 요청에는 고유한 `id`가 존재
- 각 도구를 실행한 뒤 해당 ID와 연결된 별도의 `ToolMessage`를 생성
- 여러 도구의 실행 결과를 LLM에 전달하여 최종 응답을 생성

순서

1. `for`문으로 각 Tool Call을 순회
2. 각 도구를 `invoke()`로 실제 실행
3. 실행 결과를 각각 `ToolMessage`로 만들어 `messages`에 추가
4. `model_with_tools.invoke(messages)`를 호출하여 LLM에 실행 결과를 전달

예시 코드

```python
# 도구 이름과 실제 Tool 객체의 매핑
tool_map = {
    "get_weather": get_weather,
    "get_population": get_population
}

for tc in response.tool_calls:
    selected_tool = tool_map[tc["name"]]
    result = selected_tool.invoke(tc["args"])

    messages.append(
        ToolMessage(
            content=str(result),
            tool_call_id=tc["id"]
        )
    )
```

결국 정리하면 LLM이 한 번에 여러 개의 도구 호출을 할 수 있음.

→ 이 경우 해당 도구 호출을 각각 실행한 뒤, 실행 결과를 별도의 `ToolMessage`로 생성

→ `tool_call_id`를 통해 각 실행 결과와 원래 도구 호출 요청을 연결(LLM이 어떤 호출이 어떤 결과가 나왔는지 알 수 있도록)

+) 반복하지만, `messages.append()`는 메시지를 대화 기록에 추가할 뿐(ex. 도구 실행 결과 추가) 결과를 LLM이 처리하게 하려면 `model.invoke(messages)`를 다시 호출해야 한다!!!!

### 6. Error Handling

- 도구 실행 중 오류가 발생하면 `try-except` 등을 사용해 처리
- 실패 결과 역시 `ToolMessage`로 전달하여 LLM이 후속 행동(재시도, 다른 도구 선택, 오류 안내)을 결정하도록 할 수 있음
- 실제 서비스에서는 오류 메시지의 민감 정보 노출을 주의해야 함

### 7. Community Tools

- `LangChain`은 웹 검색, 위키피디아 조회 등 미리 구현된 도구를 제공
- 직접 만든 Custom Tool과 Community Tool을 함께 사용 가능

### 8. Additional Features

- Pydantic `args_schema`: 도구의 인자 설명과 제약 조건을 상세하게 정의하는 방식
- `tool_choice`: LLM의 도구 선택 방식을 제어하는 것
  - `auto`: LLM이 도구 사용 여부를 판단
  - `required`: 반드시 도구를 호출하도록 지정
  - `none`: 도구 사용을 비활성화
