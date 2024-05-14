from collections import OrderedDict

class URLCache:
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.cache = OrderedDict()

    def is_cached(self, url):
        """주어진 URL이 캐시에 있는지 확인하고, 있으면 최근 사용된 것으로 갱신합니다."""
        if url in self.cache:
            self.cache.move_to_end(url)  # 최근 사용된 항목으로 이동
            return True
        return False

    def add_to_cache(self, url):
        """URL을 캐시에 추가합니다. 캐시가 가득 차면 가장 오래된 URL을 제거합니다."""
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)  # OrderedDict에서 가장 오래된 항목을 제거
        self.cache[url] = True  # URL을 추가하고 최근 사용된 항목으로 마킹
