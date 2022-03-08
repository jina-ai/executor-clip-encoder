from pathlib import Path
from typing import Tuple

import clip
import numpy as np
import pytest
import torch
from PIL import Image
from jina import Document, DocumentArray, Executor

from executor import CLIPEncoder


@pytest.fixture(scope='module')
def encoder() -> CLIPEncoder:
    return CLIPEncoder()


@pytest.fixture(scope='module')
def encoder_no_preprocess() -> CLIPEncoder:
    return CLIPEncoder(use_default_preprocessing=False)


@pytest.fixture(scope='function')
def nested_docs() -> DocumentArray:
    tensor = np.ones((224, 224, 3), dtype=np.uint8)
    docs = DocumentArray([Document(id='root1', tensor=tensor)])
    docs[0].chunks = [
        Document(id='chunk11', tensor=tensor),
        Document(id='chunk12', tensor=tensor),
        Document(id='chunk13', tensor=tensor),
    ]
    docs[0].chunks[0].chunks = [
        Document(id='chunk111', tensor=tensor),
        Document(id='chunk112', tensor=tensor),
    ]
    return docs


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.batch_size == 32
    assert ex.pretrained_model_name_or_path == 'openai/clip-vit-base-patch32'


def test_no_documents(encoder: CLIPEncoder):
    docs = DocumentArray()
    encoder.encode(docs=docs, parameters={})
    assert len(docs) == 0


def test_docs_no_content(encoder: CLIPEncoder):
    docs = DocumentArray([Document()])
    encoder.encode(docs=DocumentArray(), parameters={})
    assert len(docs) == 1
    assert docs[0].embedding is None


def test_single_image(encoder: CLIPEncoder):
    docs = DocumentArray([Document(tensor=np.ones((100, 100, 3), dtype=np.uint8))])
    encoder.encode(docs, {})
    assert docs[0].embedding.shape == (512,)
    assert docs[0].embedding.dtype == np.float32


def test_single_image_no_preprocessing(encoder_no_preprocess: CLIPEncoder):
    docs = DocumentArray([Document(tensor=np.ones((3, 224, 224), dtype=np.uint8))])
    encoder_no_preprocess.encode(docs, {})
    assert docs[0].embedding.shape == (512,)
    assert docs[0].embedding.dtype == np.float32


def test_single_text(encoder: CLIPEncoder):
    docs = DocumentArray([Document(text='some text here')])
    encoder.encode(docs, {})
    assert docs[0].embedding.shape == (512,)
    assert docs[0].embedding.dtype == np.float32


def test_encoding_cpu():
    encoder = CLIPEncoder(device='cpu')
    docs = DocumentArray(
        [
            Document(text='some text'),
            Document(tensor=np.ones((100, 100, 3), dtype=np.uint8))
        ]
    )
    encoder.encode(docs=docs, parameters={})
    assert docs[0].embedding.shape == (512,)
    assert docs[1].embedding.shape == (512,)


def test_encoding_cpu_no_preprocessing():
    encoder = CLIPEncoder(device='cpu', use_default_preprocessing=False)
    docs = DocumentArray(
        [
            Document(text='some text'),
            Document(tensor=np.ones((3, 224, 224), dtype=np.uint8))
        ]
    )
    encoder.encode(docs=docs, parameters={})
    assert docs[0].embedding.shape == (512,)
    assert docs[1].embedding.shape == (512,)


@pytest.mark.gpu
def test_encoding_gpu():
    encoder = CLIPEncoder(device='cuda')
    docs = DocumentArray(
        [
            Document(text='some text'),
            Document(tensor=np.ones((100, 100, 3), dtype=np.uint8))
        ]
    )
    encoder.encode(docs=docs, parameters={})
    assert docs[0].embedding.shape == (512,)
    assert docs[1].embedding.shape == (512,)


@pytest.mark.gpu
def test_encoding_gpu_no_preprocessing():
    encoder = CLIPEncoder(device='cuda', use_default_preprocessing=False)
    docs = DocumentArray(
        [
            Document(text='some text'),
            Document(tensor=np.ones((3, 224, 224), dtype=np.uint8))
        ]
    )
    encoder.encode(docs=docs, parameters={})
    assert docs[0].embedding.shape == (512,)
    assert docs[1].embedding.shape == (512,)


def test_any_image_shape(encoder: CLIPEncoder):
    docs = DocumentArray([Document(tensor=np.ones((224, 224, 3), dtype=np.uint8))])
    encoder.encode(docs=docs, parameters={})
    assert len(docs.embeddings) == 1

    docs = DocumentArray([Document(tensor=np.ones((100, 100, 3), dtype=np.uint8))])
    encoder.encode(docs=docs, parameters={})
    assert len(docs.embeddings) == 1


def test_batch(encoder: CLIPEncoder):
    docs = DocumentArray(
        [
            Document(tensor=np.ones((100, 100, 3), dtype=np.uint8)),
            Document(tensor=np.ones((100, 100, 3), dtype=np.uint8)),
        ]
    )
    encoder.encode(docs, parameters={})
    assert len(docs.embeddings) == 2
    assert docs[0].embedding.shape == (512,)
    assert docs[0].embedding.dtype == np.float32
    np.testing.assert_allclose(docs[0].embedding, docs[1].embedding)

def test_batch_blob(encoder: CLIPEncoder):
    docs = DocumentArray(
        [
            Document(tensor=np.ones((100, 100, 3), dtype=np.uint8)).convert_image_tensor_to_blob(),
            Document(tensor=np.ones((100, 100, 3), dtype=np.uint8)).convert_image_tensor_to_blob(),
        ]
    )
    encoder.encode(docs, parameters={})
    assert len(docs.embeddings) == 2
    docs[0].convert_blob_to_image_tensor()
    assert docs[0].embedding.shape == (512,)
    assert docs[0].embedding.dtype == np.float32
    np.testing.assert_allclose(docs[0].embedding, docs[1].embedding)

def test_batch_uri(encoder: CLIPEncoder):
    docs = DocumentArray(
        [
            Document(tensor=np.ones((100, 100, 3), dtype=np.uint8)).convert_image_tensor_to_uri(),
            Document(tensor=np.ones((100, 100, 3), dtype=np.uint8)).convert_image_tensor_to_uri(),
        ]
    )
    docs[0].tensor = None
    docs[1].tensor = None
    encoder.encode(docs, parameters={})
    assert len(docs.embeddings) == 2
    assert docs[0].uri != ''
    assert docs[0].embedding.shape == (512,)
    assert docs[0].embedding.dtype == np.float32
    np.testing.assert_allclose(docs[0].embedding, docs[1].embedding)


def test_batch_no_preprocessing(encoder_no_preprocess: CLIPEncoder):
    docs = DocumentArray(
        [
            Document(tensor=np.ones((3, 224, 224), dtype=np.float32)),
            Document(tensor=np.ones((3, 224, 224), dtype=np.float32)),
        ]
    )
    encoder_no_preprocess.encode(docs, {})
    assert len(docs.embeddings) == 2
    assert docs[0].embedding.shape == (512,)
    assert docs[0].embedding.dtype == np.float32
    np.testing.assert_allclose(docs[0].embedding, docs[1].embedding)


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_size(encoder: CLIPEncoder, batch_size: int):
    tensor = np.ones((100, 100, 3), dtype=np.uint8)
    docs = DocumentArray([Document(tensor=tensor) for _ in range(32)])
    encoder.encode(docs, parameters={'batch_size': batch_size})
    for doc in docs:
        assert doc.embedding.shape == (512,)


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_size_no_preprocessing(
    encoder_no_preprocess: CLIPEncoder, batch_size: int
):
    tensor = np.ones((3, 224, 224), dtype=np.uint8)
    docs = DocumentArray([Document(tensor=tensor) for _ in range(32)])
    encoder_no_preprocess.encode(docs, parameters={'batch_size': batch_size})
    for doc in docs:
        assert doc.embedding.shape == (512,)


def test_overwrite_embeddings(encoder: CLIPEncoder):
    docs = DocumentArray(
        [
            Document(text='foo', embedding=np.random.rand(10))
            for _ in range(5)
        ] +
        [
            Document(
                tensor=np.random.randint(0, 255, (100, 100, 3)),
                embedding=np.random.rand(10)
            ) for _ in range(5)
        ]
    )
    encoder.encode(docs, parameters={})
    for doc in docs:
        assert doc.embedding.shape == (10,)

    encoder.encode(docs, parameters={'overwrite_embeddings': False})
    for doc in docs:
        assert doc.embedding.shape == (10,)

    encoder.encode(docs, parameters={'overwrite_embeddings': True})
    for doc in docs:
        assert doc.embedding.shape == (512,)


def test_embeddings_quality(encoder: CLIPEncoder):
    """
    This tests that the embeddings actually "make sense".
    We check this by making sure that the distance between the embeddings
    of two similar images and texts is smaller than everything else.
    """
    data_dir = Path(__file__).parent.parent / 'imgs'
    docs = DocumentArray(
        [
            Document(id='dog', tensor=np.array(Image.open(data_dir / 'dog.jpg'))),
            Document(id='cat', tensor=np.array(Image.open(data_dir / 'cat.jpg'))),
            Document(
                id='airplane', tensor=np.array(Image.open(data_dir / 'airplane.jpg'))
            ),
            Document(
                id='helicopter',
                tensor=np.array(Image.open(data_dir / 'helicopter.jpg'))
            ),
            Document(id='A', text='a furry animal that with a long tail'),
            Document(id='B', text='a domesticated mammal with four legs'),
            Document(id='C', text='a type of aircraft that uses rotating wings'),
            Document(id='D', text='flying vehicle that has fixed wings and engines'),

        ]
    )
    encoder.encode(docs, {})

    docs.match(docs)
    matches = ['cat', 'dog', 'helicopter', 'airplane', 'B', 'A', 'D', 'C']
    for i, doc in enumerate(docs):
        assert doc.matches[1].id == matches[i]


def test_openai_embed_match():
    data_dir = Path(__file__).parent.parent / 'imgs'
    image_docs = [
        Document(tensor=np.array(Image.open(data_dir / 'dog.jpg'))),
        Document(tensor=np.array(Image.open(data_dir / 'airplane.jpg'))),
        Document(tensor=np.array(Image.open(data_dir / 'helicopter.jpg'))),
    ]
    text_docs = [
        Document(text='Jina AI is lit'),
        Document(text='Jina AI is great'),
        Document(text='Jina AI is a cloud-native neural search company'),
        Document(text='Jina AI is a github repo'),
        Document(text='Jina AI is an open source neural search project'),
    ]
    docs = DocumentArray(image_docs + text_docs)

    executor = CLIPEncoder('openai/clip-vit-base-patch32')
    executor.encode(DocumentArray(docs), {})
    executor_embeddings = docs.embeddings

    model, preprocess = clip.load('ViT-B/32', device='cpu')
    with torch.no_grad():
        images = [Image.fromarray(doc.tensor) for doc in image_docs]
        tensors = [preprocess(img) for img in images]
        tensor = torch.stack(tensors)
        clip_image_embeddings = model.encode_image(tensor).numpy()

        tokens = clip.tokenize([doc.text for doc in text_docs])
        clip_text_embeddings = model.encode_text(tokens).numpy()

    clip_embeddings = np.concatenate(
        [clip_image_embeddings, clip_text_embeddings], axis=0
    )
    np.testing.assert_almost_equal(clip_embeddings, executor_embeddings, 5)


@pytest.mark.parametrize(
    'traversal_paths, counts',
    [
        ['@r', (('@r', 1), ('@c', 0), ('@cc', 0))],
        ['@c', (('@r', 0), ('@c', 3), ('@cc', 0))],
        ['@cc', (('@r', 0), ('@c', 0), ('@cc', 2))],
        ['@cc,r', (('@r', 1), ('@c', 0), ('@cc', 2))],
    ],
)
def test_traversal_path(
    traversal_paths: str,
    counts: Tuple[str, int],
    nested_docs: DocumentArray,
    encoder: CLIPEncoder,
):
    encoder.encode(nested_docs, parameters={'traversal_paths': traversal_paths})
    for path, count in counts:
        embeddings = nested_docs[path].embeddings
        if count != 0:
            assert len([em for em in embeddings if em is not None]) == count
        else:
            assert embeddings is None
