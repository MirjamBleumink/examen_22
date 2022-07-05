# sys.path.insert(0, "../..")
import trax # noqa F401
from trax import layers as tl
from trax.layers import combinators as cb
from trax.layers.assert_shape import assert_shape
from trax.shapes import signature
from typing import Union
import jax.numpy as jnp
from numpy import ndarray
from trax.layers.combinators import Serial as Traxmodel
# from trax.layers import AttentionQKV
from trax.layers import PureAttention
# from trax.shapes import signature
import gin

Array = Union[jnp.ndarray, ndarray]


def Cast():
    def f(generator):
        for x, y in generator:
            yield x.numpy(), y.numpy()

    return lambda g: f(g)


def summary(
    model: Traxmodel, X: Array, init: int = 1, counter: int = 0  # noqa N803
) -> Array:
    output = X  # noqa N803
    input = signature(output)
    if init == 1:
        print(
            f'{"layer":<23}{"input":<19} {"dtype":^7}{"output":<19} {"dtype":^7}' # noqa E501
        )
    for sub in model.sublayers:
        name = str(sub.name)
        if name == "":
            continue
        elif name == "Serial":
            output = summary(sub, output, init + 1, counter)
        else:
            output = sub.output_signature(input)
            print(
                f"({counter}) {str(sub.name):<19} {str(input.shape):<19}({str(input.dtype):^7}) | {str(output.shape):<19}({str(output.dtype):^7})"  # noqa E501
            )
        input = output
        counter += 1
    return output


@assert_shape("bs->bsd")
def last() -> None:
    return tl.Fn("Last", lambda x: x[:, -1, :], n_out=1)


def AvgLast():
    return tl.Fn("AvgLast", lambda x: x.mean(axis=-1), n_out=1)


@gin.configurable
def Trax_Basemodel(vocab_size: int, d_feature: int, d_out: int) -> None:
    model = cb.Serial(
        tl.Embedding(vocab_size=vocab_size, d_feature=d_feature),
        tl.GRU(n_units=d_feature),
        tl.BatchNorm(),
        AvgLast(),
        tl.Dense(d_out),
    )
    return model


@gin.configurable
def Trax_AttentionModel(vocab_size: int, d_feature: int, d_out: int, n_heads: int, dropout: float, mode: str) -> None: # noqa E501
    model = cb.Serial(
        tl.Embedding(vocab_size=vocab_size, d_feature=d_feature),
        tl.GRU(n_units=d_feature),
        tl.BatchNorm(),
        AvgLast(),
        tl.Dense(d_out),
        cb.Parallel(
          tl.Dense(d_feature),
          tl.Dense(d_feature),
          tl.Dense(d_feature),
        ),
        PureAttention(n_heads=n_heads, dropout=dropout, mode=mode),
        tl.Dense(d_feature),
        tl.Relu(),
        tl.Dense(d_out),
    )
    return model
