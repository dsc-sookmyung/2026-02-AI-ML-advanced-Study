# Ch.05 Tool Calling과 Function Calling 정리

> **LLM의 한계 → Tool Calling → 도구 정의 → 도구 연결 → 실행 결과 처리**
>
> 참고: https://wikidocs.net/333789

## 1. Tool Calling이란?

**Tool Calling**은 LLM이 질문을 해결하기 위해 필요한 **외부 도구(Function/API)**와 **입력 인자**를 선택하는 기능이다.

LLM은 실시간 조회, 정확한 계산, DB 접근 등을 직접 수행하지 못할 수 있으므로 도구를 활용한다.

```text
사용자 질문
    ↓
LLM: 사용할 도구와 인자 결정
    ↓
애플리케이션 코드: 실제 도구 실행
    ↓
도구 결과를 LLM에 전달
    ↓
LLM: 최종 답변 생성
```

- **Function Calling**: Tool Calling과 거의 같은 의미로 쓰이는 용어
- **핵심**: LLM은 *실행 요청*을 만들고, 실제 실행은 **개발자 코드**가 담당한다.

## 2. `@tool`로 도구 정의

LangChain에서는 Python 함수에 `@tool` 데코레이터를 붙여 LLM이 사용할 도구로 변환할 수 있다.

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """두 정수를 곱합니다."""
    return a * b
```

| 요소 | 역할 |
|---|---|
| 함수 이름 | 도구 이름 |
| Docstring | LLM이 도구의 용도를 판단하는 설명 |
| Type Hint | 인자 타입 및 스키마 생성에 활용 |
| `args_schema` | Pydantic으로 입력값과 제약 조건을 상세 정의 |

복잡한 도구는 `StructuredTool.from_function()` 또는 `BaseTool`로 만들 수도 있다.

## 3. `bind_tools()`로 모델에 연결

도구를 정의한 후 **LLM에 사용할 수 있는 도구 목록을 알려준다.**

```python
model_with_tools = model.bind_tools([multiply])
response = model_with_tools.invoke("3 곱하기 12는?")
```

- `bind_tools()`는 도구의 **이름·설명·입력 스키마**를 모델에 전달한다.
- 원본 모델을 수정하는 것이 아니라 **새 모델 객체를 반환**한다.
- LLM은 도구가 필요 없다고 판단하면 일반 답변을 생성할 수 있다.

## 4. `tool_calls`와 `ToolMessage`

LLM이 도구 호출을 결정하면 `AIMessage.tool_calls`에 요청 정보가 담긴다.

```text
AIMessage.tool_calls
 ├─ name: multiply
 ├─ args: {a: 3, b: 12}
 └─ id: 도구 호출 식별자
```

이때 **아직 도구가 실행된 것은 아니다.** 애플리케이션에서 실제 함수를 호출해야 한다.

```python
call = response.tool_calls[0]
result = multiply.invoke(call["args"])

from langchain_core.messages import ToolMessage
message = ToolMessage(
    content=str(result),
    tool_call_id=call["id"]
)
```

`ToolMessage`는 도구 실행 결과를 LLM에 다시 전달할 때 사용하며, `tool_call_id`로 요청과 결과를 연결한다.

## 5. Tool Execution Loop

```text
HumanMessage: "3 곱하기 12는?"
         ↓
LLM → AIMessage(tool_calls)
         ↓
코드 → multiply(3, 12) 실행
         ↓
ToolMessage: "36"
         ↓
LLM → "3 곱하기 12는 36입니다."
```

여러 도구를 연결하면 LLM이 질문에 따라 적합한 도구를 선택하거나 복수의 호출을 요청할 수 있다.

## 6. 내장 도구와 커뮤니티 도구

LangChain에서는 자주 사용되는 도구를 재사용할 수 있다.

| 도구 | 용도 |
|---|---|
| `DuckDuckGoSearchRun` | 웹 검색 |
| `WikipediaQueryRun` | 위키피디아 검색 |
| `PythonREPLTool` | Python 코드 실행 |
| 파일 관련 도구 | 파일 읽기 및 처리 |

> 코드 실행이나 외부 API 호출 도구는 **권한 제한, 입력 검증, 보안**을 고려해야 한다.

## 핵심 흐름

```text
LLM 단독 사용
    ↓
외부 정보 조회·작업 수행에 한계
    ↓
@tool → 사용 가능한 도구 정의
    ↓
bind_tools() → LLM에 도구 연결
    ↓
LLM → tool_calls 생성
    ↓
애플리케이션 → 도구 실제 실행
    ↓
ToolMessage → 결과 전달
    ↓
LLM → 최종 응답
```

## 핵심 개념 정리

- **Tool Calling**: LLM이 필요한 도구와 입력 인자를 결정하는 기능
- **`@tool`**: Python 함수를 LangChain 도구로 변환
- **`bind_tools()`**: 모델에 도구 정보 연결
- **`tool_calls`**: LLM이 생성한 도구 호출 요청
- **`invoke()`**: 도구를 실제로 실행
- **`ToolMessage`**: 실행 결과를 LLM에 전달하는 메시지
- **Tool Execution Loop**: 도구 요청 → 실행 → 결과 전달 → 최종 답변

---

**한 줄 요약:** Tool Calling은 **LLM이 어떤 도구를 사용할지 판단하고, 애플리케이션이 실행한 결과를 다시 LLM이 활용하게 하는 구조**이다.
