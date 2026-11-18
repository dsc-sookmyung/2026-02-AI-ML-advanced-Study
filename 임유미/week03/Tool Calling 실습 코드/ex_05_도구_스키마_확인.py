from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import os

# 도구 정의
@tool
def multiply(a: int, b: int) -> int:
    """두 정수를 곱합니다."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 조회합니다."""
    weather = {"서울": "맑음 22도", "부산": "흐림 25도", "제주": "비 19도"}
    return weather.get(city, f"{city}: 정보 없음")

# bind_tools()가 생성하는 스키마 확인 (API 키 없이)
tools = [multiply, get_weather]
print("=== 바인딩할 도구 목록 ===")
for t in tools:
    print(f"  - {t.name}: {t.description}")

# 도구 스키마를 JSON으로 변환하여 확인
import json
schemas = []
for t in tools:
    schema = {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.args_schema.model_json_schema()
        }
    }
    schemas.append(schema)

print("\n=== LLM에 전달되는 도구 스키마 ===")
print(json.dumps(schemas, indent=2, ensure_ascii=False))

# tool_choice 옵션 설명
print("\n=== tool_choice 옵션 ===")
options = {
    "auto": "LLM이 자율 판단 (기본값)",
    "required": "반드시 도구 중 하나 호출",
    "none": "도구 사용 안 함",
}
for k, v in options.items():
    print(f"  tool_choice='{k}': {v}")