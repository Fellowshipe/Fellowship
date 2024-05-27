# 중고나라 게시물을 활용한 멀티모달 모델 기반 중고거래 사기탐지

## 전처리 내용
### RDMS 데이터 전처리
- price -> int로 변환
- price -> 0의 비율
- member_level, product_status, payment_method, shipping_method -> categorical variable -> one-hot-encoding
- transaction_region -> None은 0, 데이터 있으면 1
- post_date -> 요일 변수 생성, 시간대 변수 생성(00:00 ~ 01:00 -> 0 ~ 23:00 ~ 24:00 -> 23)
- title, description -> ? (concat?) -> 토큰화 및 불용어 처리 -> CLIP, VisualBERT 입력에 적합하도록 텍스트 전처리 수행 예정
- description -> 게시글 길이 변수 생성
- description -> 특수문자 및 숫자의 비율
 
### S3 데이터 전처리
- 이미지 -> CLIP, VisualBERT 입력에 적합하도록 이미지 전처리 수행 예정
- 이미지 -> image_count 변수 추출
