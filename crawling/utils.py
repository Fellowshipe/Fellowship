# 제품 디테일 추출 함수
def extract_information(commercialDetail):
    # 정보를 추출할 태그 목록
    tags_to_find = ['상품 상태', '결제 방법', '배송 방법']

    # 결과를 저장할 딕셔너리
    results = {}

    # 'detail_list' 클래스를 가진 모든 'dl' 태그를 찾음
    all_dl = commercialDetail.find_all('dl', class_='detail_list')

    # 각 태그 목록에 대해 처리
    for tag in tags_to_find:
        # 해당 태그가 포함된 'dl' 태그 찾기
        dl = next((dl for dl in all_dl if dl.find('dt') and dl.find('dt').get_text(strip=True) == tag), None)
        
        # 해당 'dl' 태그가 있고, 그 안에 'dd' 태그도 있으면 텍스트를 가져옴
        if dl and dl.find('dd'):
            results[tag] = dl.find('dd').get_text(strip=True)
        else:
            # 해당 태그가 없으면 결과에 None 저장
            results[tag] = None

    return results