from abc import abstractmethod

class JungoNara():

    def __init__(self, delay_time=None, saving_html=False):
        """
        Args:
            delay_time (float or tuple, optional): Defalults to None.
                When 'delay_time=float', it will crawl sites with delay.
                When 'delay_time=tuple', it will crawl sites with random delay.

            saving_html (bool, optional): Defaults to False.
                When 'saving_html=False', it always requests url every function calling.
                When 'saving_html=True', It will save requested html only first time.
                After that, ti calls saved html. This will help to alleviate server load.
        """

        self.delay_time = delay_time
        self.saving_html = saving_html

    def dynamic_crawl(self, url: str | list[str]) -> list[str]:

        """
        Return article text using Selenium.

        Args:
            url (str | list):
                when 'url=str', it will only crawl given url.
                When 'url=list' it will crawl with iterating url list.

        Returns:
            list: Return article list.
        """

        if type(url) == str:
            return [self._dynamic_crawl(url)]
        elif type(url) == list:
            return [self._dynamic_crawl(url_str) for url_str in url]
        else: raise TypeError("You must give url string or list type.")

    @abstractmethod
    def _dynamic_crawl(self, url: str) -> str:
        pass

    def static_crawl(self, url: str | list[str]) -> list[str]:

        """
        Return article text using Selenium.

        Args:
            url (str | list):
                when 'url=str', it will only crawl given url.
                When 'url=list' it will crawl with iterating url list.

        Returns:
            list: Return article list.
        """

        if type(url) == str:
            return [self._static_crawl(url)]
        elif type(url) == list:
            return [self._static_crawl(url_str) for url_str in url]
        else: raise TypeError("You must give url string or list type.")

    @abstractmethod
    def _static_crawl(self, html: str) -> str:
        pass

    @abstractmethod
    def _parse_html(self, html: str) -> str:
        pass
