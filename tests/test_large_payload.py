from io import BytesIO

import numpy as np
import pytest

from matio.v5._mio5_utils import VarReader5
from matio.v5.matio5_params import mxTypes
from matio.v5.matreader5 import MatFile5Reader
from matio.v5.matwriter5 import MatFile5Writer, VarWriter5


@pytest.mark.parametrize("count", [3, 2**28 - 1])
def test_large_matrix_header(count):
    # Reading a header must not require allocating its numeric payload.
    stream = BytesIO()
    writer = VarWriter5(MatFile5Writer(stream, oned_as="col"))
    writer._var_name = np.asarray(b"signal")
    writer.write_header((count, 1), mxTypes.mxDOUBLE_CLASS)
    header_end = stream.tell()
    data_bytes = count * 8
    payload_size = header_end - 8 + 8 + data_bytes

    stream.seek(8)
    reader = VarReader5(MatFile5Reader(stream))
    header = reader.read_header(False, payload_size)
    assert header.name == b"signal"
    assert header.dims == [count, 1]
    assert header.mclass == mxTypes.mxDOUBLE_CLASS
    assert header.payload_size == data_bytes + 8
    assert stream.tell() == header_end
