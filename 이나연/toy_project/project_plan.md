## 👤 제출자 정보

- **이름**: 이나연
- **팀 / 개인 여부**: 개인

---

## 📝 프로젝트 개요

### 프로젝트명

**FastReAct** — KV Cache 재사용 기반 고속 응답형 ReAct 에이전트 추론 엔진

### 한 줄 소개

멀티턴 ReAct 에이전트의 도구 호출 루프에서 KV Cache를 재사용(Prefix Cache + 증분 디코딩)하여 TTFT와 VRAM 사용량을 줄이는 경량 추론 파이프라인

### 기획 배경 및 문제 정의

- ReAct 에이전트는 `Thought → Action → Observation` 을 반복하며, 턴이 지날수록 프롬프트(시스템 프롬프트 + 툴 스키마 + 이전 대화/관찰)가 계속 길어집니다.
- 바닐라 방식은 매 턴마다 **전체 컨텍스트를 처음부터 다시 prefill** 하기 때문에, 턴 수가 늘수록 TTFT(Time To First Token)가 선형 이상으로 증가하고 불필요한 연산이 반복됩니다.
- 하지만 ReAct 컨텍스트는 **append-only 구조**(앞부분은 그대로 두고 뒤에 새 내용만 붙음)라서, 이전 턴까지의 KV Cache를 그대로 재사용하고 **새로 추가된 토큰(Observation 등)만 증분 prefill** 하는 방식을 구현해보고자 합니다.
- 특히 시스템/툴 프롬프트는 모든 요청에서 동일한 고정 prefix이므로 한 번만 계산해 두고 재사용할 수 있습니다.
- 또한 툴 호출이 실패했을 경우에도, 잘못된 Action/Observation 구간을 KV Cache에서 잘라내고(rollback) 이전 상태에서 재시도하면, 전체를 다시 계산하지 않고도 복구할 수 있다.
- 따라서, 에이전트 추론 최적화(Inference Optimization)에 관심이 있어 이 문제를 직접 구현하고 수치로 검증해보고자 선택했습니다.

---

## 🎯 목표

- **핵심 목표**: 경량 LLM 기반 ReAct 루프에 KV Cache 재사용을 적용하여, 바닐라 방식 대비 **턴별 TTFT 및 VRAM 사용량이 얼마나 줄어드는지 정량적으로 검증**합니다. (출력 품질은 동일하게 유지합니다)
- **최소 구현 범위 (MVP)**:
    1. Llama-3.2 기반 ReAct 루프 구현 (바닐라 방식, 매 턴 전체 재계산)
    2. 시스템/툴 프롬프트 Prefix Cache 고정 + 이전 턴 KV Cache를 유지하는 증분 추론 구현
    3. 바닐라 vs 캐시 재사용 방식의 TTFT / 턴별 latency / Peak VRAM 벤치마크 및 결과 정리
- **추가로 시도해보고 싶은 것 (Optional)**: 툴 호출 실패(파싱 오류, 툴 에러) 시 **캐시 스냅샷 롤백** 구현 (cache crop 활용)

---

## 👉 태스크 & 접근 방법

- **태스크 유형**: <u>에이전트 추론 최적화</u> 
(ReAct 기반 툴 사용 / 멀티스텝 질의응답 — 모델 성능 개선이 아닌 추론 속도·메모리 효율 개선이 중심)
- **사용할 모델 (후보)**: Llama-3.2-3B-Instruct (기본), Llama-3.2-1B-Instruct (디버깅용)
- **접근 방법**: 
학습/파인튜닝 없이 **추론 파이프라인 구현** 중심
    1. **Baseline**: Hugging Face Transformers로 매 턴 전체 프롬프트를 재토크나이즈 후 처음부터 `generate()`
    2. **Prefix Cache**: 여러 요청/세션이 공유하는 시스템 프롬프트 + 툴 스키마의 KV Cache를 한 번만 계산해 고정하고, 새 세션을 시작할 때마다 복제하여 재사용
    3. **증분 추론**: 한 세션 내에서 턴이 진행될 때 이전 턴까지의 `past_key_values`를 유지하고, 새로 추가된 Observation 토큰만 forward 하여 이어서 생성
    - 토큰화 경계 문제를 피하기 위해 텍스트 재토크나이즈 대신 토큰 ID를 직접 이어붙이는 방식으로 관리
    - chat template의 특수 토큰 처리 검증 필요 (BOS 중복 방지, 시스템 프롬프트의 날짜 문자열 고정 등)

*  벤치마크에서는 세션 시작 구간(Prefix Cache 효과)과 턴 진행 구간(증분 추론 효과)을 구분하여 측정

- **선택 이유**:
    - Llama-3.2는 1B/3B가 동일한 아키텍처와 토크나이저를 공유하므로, 1B로 캐시 로직을 디버깅한 뒤 코드 변경 없이 3B로 벤치마크할 수 있습니다. 또한 표준 full attention 구조라 캐시 조작이 단순합니다.
    - 소형 모델은 단일 GPU(Colab T4 등)에서도 실험 가능하여 토이프로젝트 일정에 적합합니다.
    - ReAct 컨텍스트가 append-only라 캐시 재사용 효과가 명확하게 나타나는 구조라서, 최적화 전후 차이를 수치로 보여주기 좋습니다.
    - KV Cache, Prefix Caching, 증분 디코딩 등 LLM 서빙의 핵심 개념을 직접 구현하며 학습할 수 있다는 점이 가장 큰 이유입니다.

---

## 📊 데이터

| 항목 | 내용 |
| --- | --- |
| 데이터셋 이름 / 출처 | 벤치마크용 에이전트 태스크 질문셋: **HotpotQA** 일부 샘플 + **GSM8K** 일부 샘플. 
툴은 직접 구현한 mock 도구(검색, 계산기 등)로 구성 |
| 데이터 규모 | 각 50~100개 질문 샘플 (턴 수별 비교를 위해 2턴 / 5턴 / 10턴 시나리오로 분류) |
| 언어 | 영어 |
| 라이선스 / 사용 가능 여부 | HotpotQA: CC BY-SA 4.0, GSM8K: MIT — 연구/학습 목적 사용 가능 (사용 전 각 데이터셋 카드에서 재확인이 필요). |
| 전처리 계획 | 질문/정답 쌍 샘플링 → 필요한 턴 수를 유도하도록 시나리오 분류 → mock 툴(검색은 HotpotQA 지원 문서 기반 로컬 검색, 계산기는 Python eval 래퍼) 구성 → ReAct 프롬프트 템플릿 및 툴 스키마 고정 |

---

## 📏 평가 방법

- **평가 지표**:
    - **TTFT (Time To First Token)**: 턴별 측정 (턴 번호에 따른 증가 추이 비교)
    - **턴별 / 전체 latency** 및 tokens/s
    - **Peak VRAM** (`torch.cuda.max_memory_allocated`)
    - **정확도/동일성 검증**: 캐시 재사용 방식과 바닐라 방식의 출력이 동일한지(greedy 디코딩 기준 토큰 일치율), 최종 답변 정확도(EM) 비교 — 속도를 얻는 대신 품질이 깨지지 않았음을 확인
- **베이스라인**: 매 턴 전체 프롬프트를 다시 prefill 하는 바닐라 ReAct 루프 (동일 모델, 동일 디코딩 설정)

---

## 🛠️ 기술 스택

- **언어 / 프레임워크**: Python, PyTorch, Hugging Face Transformers
- **실험 환경**: Google Colab (T4/L4 GPU)
- **(선택) 서빙 / 데모**: Gradio 또는 Streamlit으로 바닐라 vs 캐시 재사용 응답 속도 비교 데모

---

## 🗓️ 일정 계획

| 주차 | 계획 |
| --- | --- |
| 1주차 | 개념 학습 (KV Cache, Prefix Caching, 증분 디코딩, ReAct) → Llama-3.2 환경 세팅 (Hugging Face 접근 승인 신청 포함) → **바닐라 ReAct 루프 + mock 툴 구현** |
| 2주차 | 벤치마크 측정 코드(TTFT, VRAM) 작성 → **베이스라인 수치 확보** → Prefix Cache 고정 구현 착수 |
| 3주차 | **증분 추론 구현** → 바닐라 대비 출력 동일성 검증 (토큰 일치) → 디버깅 (토큰 경계, position id, attention mask 등)  |
| 4주차 | **벤치마크 실행 및 결과 분석** (턴 수별 TTFT / VRAM 그래프) → README 및 발표 자료 정리 → (여유 시) 에러 발생 시 캐시 스냅샷 롤백 추가 |

---

## ⚠️ 예상 어려움 & 도움이 필요한 부분

- **개념 이해**: KV Cache 내부 구조, `past_key_values` / `DynamicCache` 다루기, 증분 디코딩 시 attention mask 및 position id 처리 등 처음 접하는 부분이 많아 학습 시간이 필요할 것으로 예상됨
- **토큰화 경계 문제**: 텍스트를 이어붙여 재토크나이즈하면 이전 토큰과 경계가 달라져 캐시와 불일치가 생길 수 있음 → 토큰 ID 단위 관리로 해결 예정이나 검증이 필요함
- **공정한 벤치마크 설계**: GPU warm-up, 측정 편차, 디코딩 설정 통일 등 측정 신뢰도 확보 방법에 대한 피드백을 받아야 함.
- **팀 연계**: 다른 팀원이 구축하는 에이전트와 연결한다면, 인터페이스(API 형태, 프롬프트 포맷) 협의가 필요함.

---

## 🔗 참고 자료

- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022)
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM, 2023)
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention, 2023)
- Hugging Face Transformers 문서: KV Cache / Cache classes (`DynamicCache`)
- vLLM 문서: Automatic Prefix Caching
- Llama 3.2 모델 카드 (Hugging Face meta-llama/Llama-3.2-3B-Instruct)
- HotpotQA, GSM8K 데이터셋 (Hugging Face datasets)