from jina import Executor, DocumentArray, requests


class CLIPEncoder(Executor):
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass
