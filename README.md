# violet.buddy
생성AI/LLM 교육 레포









## 메모


### TODO
- ✅ 들어가면 인사부터 하는거
- ❄️ 데이터 정형화
  - LLM에서 데이터를 알아듣는 정도
  - 평문 < 구분자로 구분된 평문 < 테이블 < JSON
- ✅ 벡터DB 생성하여 데이터 넣기
- ✅ 벡터DB에서 가장 유사한 구문을 추출
- 🟦 데이터 정형화
  - 🟦 (1) 문서를 #으로 구분하여 넣기
  - 🟦 (2) # 안에 번호가 있으면 이걸로도 구분하기
  - 🟦 (3) 표도 구분할수 있으면 따로 구분하기


### 벡터디비 호출 팁
최대한 챗지 활용하기
같은 함수에 대해서 디스크립션 여러개, 함수이름 여러개를 시도해보자?
함수를 여러개를 넣어서 여러개를 찾도록...?


### 궁금한거
크로마에 넣어서 임베딩을 생성할 때 뭐를 벡터로 만드는가?
예를들어 k 드라마 정보라고 하면 document인가?
그러면 이거는 내용을 벡터로 만드는건가?
근데 문서 넣을때 "id"는 무엇으로 할까?


### 한번에 파일 포맷팅
black .; isort .; mypy .


### 파일 실행할때
- local에서 가상환경 켜기
  - conda activate ai_llm_edu
  - (개인 맥북에서만 해당됨)
- Project1
  - python3 llm_violet/Project1/main.py
    - 또는 그냥 run 버튼으로 실행
- Project2
  - cd llm_violet/Project2
  - uvicorn main:app --host 0.0.0.0 --port 8080
- Project3
  - cd llm_violet/Project3
  - uvicorn main:app --host 0.0.0.0 --port 8080
