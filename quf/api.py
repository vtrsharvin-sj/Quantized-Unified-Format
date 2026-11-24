"""Python API: ``quf.load`` and the ``quf.Model`` handle.

Example::

    import quf

    with quf.load("model.quf") as model:
        print(model.generate("The capital of France is", max_new_tokens=24))

The handle runs the same steps as ``quf run``: QUFModel.load, the tokenizer built from the
tokenizer.json embedded in the file, and quf.runtime.generate.generate. ``generate`` returns
the text that ``quf run`` prints, without the final newline.

This module imports torch, tokenizers, and quf.runtime only inside functions, so
``import quf`` stays light.
"""

from __future__ import annotations

import errno
import os
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover
    from quf.format.schema import QUFHeader
    from quf.runtime.model import QUFModel
    from quf.tokenization.wrapper import TokenizerWrapper

__all__ = ["BACKENDS", "DEVICES", "Model", "load"]

DEVICES = ("auto", "cpu", "cuda")  # the choices of quf run --device
BACKENDS = ("auto", "cached", "fused", "marlin", "int4")  # the choices of quf run --backend


def load(
    path: str | os.PathLike[str],
    device: str = "auto",
    backend: str = "auto",
    cuda_graph: bool | None = None,
) -> Model:
    """Load a .quf file and the tokenizer embedded in it.

    Args:
        path: the .quf file.
        device: ``"auto"`` (the best Backend for this machine), ``"cpu"``, or ``"cuda"``,
            as ``quf run --device``.
        backend: ``"auto"``, ``"cached"``, ``"fused"``, ``"marlin"``, or ``"int4"``, as
            ``quf run --backend``. ``Model.backend`` names the Backend that was selected,
            for example ``"triton_int4"`` for ``"int4"``.
        cuda_graph: None (the default) replays a CUDA graph while decoding on a GPU unless
            QUF_NO_CUDA_GRAPH is set. False decodes eagerly, as ``quf run --no-cuda-graph``.

    Returns:
        A :class:`Model`. Call ``close()``, or use it in a ``with`` block, to release it.

    Raises:
        ValueError: ``device`` or ``backend`` is not an accepted value, or the file has no
            embedded tokenizer.json.
        FileNotFoundError: ``path`` does not exist.
        IsADirectoryError: ``path`` is a directory.
        quf.format.errors.QUFFormatError: the file is not a valid .quf file.
        quf.runtime.selector.NoGPUError: a GPU is required and none is available.
        quf.runtime.errors.GPUMemoryError: the model does not fit in GPU memory.

    Example::

        import quf

        model = quf.load("model.quf", device="cpu")
        print(model.backend)  # cpu
        model.close()
    """
    if device not in DEVICES:
        raise ValueError(f"device must be one of {', '.join(DEVICES)}; got {device!r}")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}; got {backend!r}")
    path = os.fspath(path)
    if os.path.isdir(path):
        raise IsADirectoryError(errno.EISDIR, "expected a .quf file but got a directory", path)
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, ".quf file not found", path)

    from quf.runtime.model import QUFModel  # noqa: PLC0415
    from quf.tokenization.wrapper import tokenizer_from_header  # noqa: PLC0415

    model = QUFModel.load(path, device_option=device, backend_option=backend, cuda_graph=cuda_graph)
    try:
        tokenizer = tokenizer_from_header(model.header)
    except BaseException:
        del model  # do not keep the weights alive through the traceback
        raise
    return Model(model, tokenizer, cuda_graph=cuda_graph)


class Model:
    """A loaded .quf model and its tokenizer. Create one with :func:`quf.load`.

    Attributes:
        tokenizer: the Tokenizer_Wrapper built from the tokenizer.json in the file
            (None after ``close()``).
        backend: the selected Backend: ``cpu``, ``torch_gpu``, ``torch_fused``, ``marlin``,
            or ``triton_int4``.
        header: the parsed header of the file (model config, tokenizer, quantization).

    Example::

        import quf

        with quf.load("model.quf") as model:
            text = model.generate("The capital of France is", max_new_tokens=24)
            model.generate("Once upon a time", stream=lambda piece: print(piece, end=""))
    """

    def __init__(
        self,
        model: QUFModel,
        tokenizer: TokenizerWrapper,
        *,
        cuda_graph: bool | None = None,
    ) -> None:
        self._model: QUFModel | None = model
        self._cuda_graph = cuda_graph
        self.tokenizer: TokenizerWrapper | None = tokenizer
        self.backend: str = model.backend_name
        self.header: QUFHeader = model.header

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int = 0,
        stream: Callable[[str], None] | None = None,
    ) -> str:
        """Continue ``prompt`` and return the generated text.

        The arguments and their limits are those of ``quf run``. Temperature 0 is greedy
        decoding; above 0, ``top_k``, ``top_p``, and ``seed`` control sampling. Generation
        stops after ``max_new_tokens`` tokens or at the end-of-sequence token, which is not
        part of the text. This is plain completion: no chat template is applied.

        Args:
            prompt: non-empty text.
            max_new_tokens: at least 1.
            temperature: 0 or more.
            top_k: None, or at least 1.
            top_p: None, or greater than 0 and at most 1.
            seed: the sampling seed.
            stream: optional callable that receives each new piece of text as soon as it
                is decoded. The pieces concatenate to the returned text.

        Returns:
            The text that ``quf run`` prints for the same options, without its final newline.

        Raises:
            ValueError: the model is closed, or an argument is outside its limits.
            quf.runtime.generate.ContextLengthError: the prompt length plus
                ``max_new_tokens`` exceeds the context length of the model.

        Example::

            text = model.generate("The capital of France is", max_new_tokens=24)
            model.generate("Once upon a time", stream=lambda piece: print(piece, end=""))
        """
        model, tokenizer = self._model, self.tokenizer
        if model is None or tokenizer is None:
            raise ValueError("the model is closed")
        _check_generation_args(prompt, max_new_tokens, temperature, top_k, top_p, stream)
        from quf.runtime.generate import generate  # noqa: PLC0415

        pieces: list[str] = []

        def collect(piece: str) -> None:
            pieces.append(piece)
            if stream is not None:
                stream(piece)

        # Always stream, as `quf run` does by default: the end-of-sequence token is left out
        # and an incomplete trailing multi-byte character is held back in the same way.
        generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            stream=collect,
            cuda_graph=self._cuda_graph,
        )
        return "".join(pieces)

    def close(self) -> None:
        """Release the model and the tokenizer. Calling it again does nothing.

        For a model on the GPU this also empties the CUDA memory cache.
        """
        model, self._model = self._model, None
        self.tokenizer = None
        if model is None:
            return
        on_cuda = getattr(model.device, "type", None) == "cuda"
        del model
        if on_cuda:
            import gc  # noqa: PLC0415

            import torch  # noqa: PLC0415

            gc.collect()
            torch.cuda.empty_cache()

    def __enter__(self) -> Model:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _check_generation_args(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    stream: Callable[[str], None] | None,
) -> None:
    """The limits of ``quf run`` (quf.cli._check_generation_args), named by argument."""
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be a str, not {type(prompt).__name__}")
    if not prompt.strip():
        raise ValueError("prompt is empty; pass some text")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1; got {max_new_tokens!r}")
    if temperature < 0:
        raise ValueError(f"temperature must not be negative; got {temperature!r}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be at least 1; got {top_k!r}")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError(f"top_p must be greater than 0 and at most 1; got {top_p!r}")
    if stream is not None and not callable(stream):
        raise TypeError("stream must be a callable that takes one str, or None")
