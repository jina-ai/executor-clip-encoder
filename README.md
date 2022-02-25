# CLIPEncoder

Encoder that embeds documents using either the CLIP vision encoder or the CLIP text encoder, depending on the content type of the document.

## Overview

**CLIPEncoder** is an encoder that wraps the image and text embedding functionality of the [CLIP](https://huggingface.co/transformers/model_doc/clip.html) model from huggingface transformers.
The encoder embeds documents using either the text or the visual part of CLIP, depending on their content.

For more information on the `gpu` usage and `volume` mounting, please refer to the [documentation](https://docs.jina.ai/tutorials/gpu-executor/).
For more information on CLIP model, checkout the [blog post](https://openai.com/blog/clip/),
[paper](https://arxiv.org/abs/2103.00020) and [hugging face documentation](https://huggingface.co/transformers/model_doc/clip.html)

## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://CLIPEncoder')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://CLIPEncoder')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
