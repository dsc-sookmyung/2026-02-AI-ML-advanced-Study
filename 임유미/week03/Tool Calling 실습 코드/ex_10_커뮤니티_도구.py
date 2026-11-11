import sys; sys.stdout.reconfigure(encoding="utf-8")

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# --- DuckDuckGo 웹 검색 도구 (API 키 불필요!) ---
web_search = DuckDuckGoSearchRun(
    name="web_search",
    description="웹에서 정보를 검색합니다. 최신 정보나 실시간 데이터가 필요할 때 사용합니다.",
)

# --- Wikipedia 검색 도구 ---
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200, lang="ko")
wikipedia_search = WikipediaQueryRun(
    name="wikipedia_search",
    description="위키피디아에서 정보를 검색합니다. 백과사전적 지식이 필요할 때 사용합니다.",
    api_wrapper=wiki_api,
)

# 도구 등록
tools = [web_search, wikipedia_search]

print("=== 등록된 커뮤니티 도구 ===")
for t in tools:
    print(f"\n도구: {t.name}")
    print(f"  설명: {t.description}")
    print(f"  인자: {list(t.args.keys())}")

# 도구 실행
print("\n=== 도구 실행 테스트 ===")
print(f"웹 검색: {web_search.invoke('LangChain framework')[:80]}...")
print(f"위키: {wikipedia_search.invoke('인공지능')[:80]}...")
