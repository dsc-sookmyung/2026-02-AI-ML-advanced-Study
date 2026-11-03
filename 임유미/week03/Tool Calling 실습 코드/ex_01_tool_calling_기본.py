from langchain_core.tools import tool # langchain_core.tools에서 @tool 데코레이터를 가져옴

# 1단계: @tool 데코레이터로 도구 정의
@tool #@tool 데코레이터로 일반 함수를 LangChain 도구로 변환
def multiply(a: int, b: int) -> int: # 함수의 타입 힌트(a: int, b: int)가 인자 스키마가 됨
    """두 정수를 곱합니다.""" # 독스트링이 도구의 설명(description)이 됩니다. LLM은 이 설명을 보고 도구를 선택
    return a * b

@tool
def add(a: int, b: int) -> int:
    """두 정수를 더합니다."""
    return a + b

# 도구의 이름과 설명 확인
print(f"도구 이름: {multiply.name}") # 도구의 name 속성은 함수 이름에서 자동 추출
print(f"도구 설명: {multiply.description}")
print(f"인자 스키마: {multiply.args_schema.model_json_schema()}") # args_schema는 Pydantic 모델로, JSON Schema 형태의 인자 정의를 제공

# 도구 직접 실행 테스트
result = multiply.invoke({"a": 3, "b": 12}) # 도구를 직접 invoke()로 호출하는 것도 가능,  LLM 없이 테스트할 때 유용
print(f"\n도구 직접 실행 결과: {result}")
print(f"결과 타입: {type(result)}")
