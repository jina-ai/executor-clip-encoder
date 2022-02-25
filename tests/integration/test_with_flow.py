import subprocess

import numpy as np
import pytest
from executor import CLIPEncoder
from jina import Document, DocumentArray, Flow


@pytest.mark.parametrize('request_size', [1])
def test_integration(request_size: int):
    docs = DocumentArray(
        [
            Document(tensor=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            for _ in range(50)
        ] + [
            Document(text='just some random text here') for _ in range(50)
        ]
    )
    with Flow().add(uses=CLIPEncoder) as flow:
        da = flow.post(on='/index', inputs=docs, request_size=request_size)

    assert len(da) == 50
    for doc in da:
        assert doc.embedding is not None
        assert doc.embedding.shape == (512,)


@pytest.mark.docker
def test_docker_runtime(build_docker_image: str):
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            ['jina', 'executor', f'--uses=docker://{build_docker_image}'],
            timeout=30,
            check=True,
        )


@pytest.mark.gpu
@pytest.mark.docker
def test_docker_runtime_gpu(build_docker_image_gpu: str):
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [
                'jina',
                'executor',
                f'--uses=docker://{build_docker_image_gpu}',
                '--gpus',
                'all',
                '--uses-with',
                'device:cuda',
            ],
            timeout=30,
            check=True,
        )
