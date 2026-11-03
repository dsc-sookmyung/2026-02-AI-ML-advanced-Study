from langchain_core.tools import tool

@tool
def search_weather(city: str, unit: str = "celsius") -> str:
    """주어진 도시의 현재 날씨를 조회합니다. # 독스트링 첫 줄이 도구의 descriptiond이 됨, LLM이 가장 먼저 보는 정보

    Args: #  Args 섹션의 설명은 JSON Schema에 포함되어 LLM이 인자를 채울 때 참고
        city: 날씨를 조회할 도시 이름 (예: '서울', 'New York')
        unit: 온도 단위. 'celsius' 또는 'fahrenheit' # 기본값이 있는 인자(unit='celsius')는 required에서 제외
    """
    # 실제로는 날씨 API를 호출합니다
    weather_data = {"서울": 22, "부산": 25, "제주": 27}
    temp = weather_data.get(city, 20)
    return f"{city}의 현재 온도: {temp}도 ({unit})"

# 도구 메타데이터 확인
print("=== 도구 메타데이터 ===")
print(f"이름(name): {search_weather.name}")
print(f"설명(description): {search_weather.description}") # name 속성은 함수 이름에서 자동 추출
print(f"반환 직접 여부(return_direct): {search_weather.return_direct}") # return_direct=False면 도구 결과를 LLM에게 돌려보내고, True면 바로 사용자에게 반환

# 인자 스키마 확인
print("\n=== 인자 스키마 (JSON Schema) ===")
import json
schema = search_weather.args_schema.model_json_schema() # model_json_schema()로 Pydantic 모델의 JSON Schema를 확인 가능
print(json.dumps(schema, indent=2, ensure_ascii=False))

# 도구 직접 실행
print("\n=== 도구 실행 ===")
result = search_weather.invoke({"city": "서울"})
print(f"결과: {result}")

result2 = search_weather.invoke({"city": "부산", "unit": "fahrenheit"})
print(f"결과: {result2}")
