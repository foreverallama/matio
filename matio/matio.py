"""Base class for MAT-file reading and writing"""

from scipy.sparse import coo_matrix, issparse

from matio.utils.matheaders import MAT_5_VERSION, MAT_HDF_VERSION, read_mat_header
from matio.v5 import loadmat5, savemat5
from matio.v7 import loadmat7, savemat7


def load_from_mat(
    file_path,
    mdict=None,
    variable_names=None,
    raw_data=False,
    add_table_attrs=False,
    spmatrix=True,
):
    """Loads variables from MAT-file"""

    subsystem_offset, ver, byte_order = read_mat_header(file_path)

    if isinstance(variable_names, str):
        variable_names = [variable_names]
    elif variable_names is not None:
        variable_names = list(variable_names)

    if ver == MAT_5_VERSION:
        matfile_dict = loadmat5(
            file_path,
            subsystem_offset,
            byte_order,
            variable_names,
            raw_data,
            add_table_attrs,
        )
    elif ver == MAT_HDF_VERSION:
        matfile_dict = loadmat7(
            file_path, byte_order, variable_names, raw_data, add_table_attrs
        )

    if len(matfile_dict["__globals__"]) == 0:
        del matfile_dict["__globals__"]

    if spmatrix:
        for name, var in list(matfile_dict.items()):
            if issparse(var):
                matfile_dict[name] = coo_matrix(var)

    # Update mdict if present
    if mdict is not None:
        mdict.update(matfile_dict)
    else:
        mdict = matfile_dict

    return mdict


def save_to_mat(
    file_path,
    mdict,
    version="v7",
    global_vars=None,
    oned_as="col",
    do_compression=True,
):
    """Saves variables to MAT-file"""

    if isinstance(global_vars, str):
        global_vars = [global_vars]
    elif global_vars is None:
        global_vars = []
    elif not isinstance(global_vars, list):
        raise ValueError("global_vars must be a list of strings")

    # subsys_offset = None

    if version == "v7":
        savemat5(file_path, mdict, global_vars, oned_as, do_compression)
    elif version == "v7.3":
        savemat7(file_path, mdict, global_vars, oned_as)
    else:
        raise ValueError(f"Unknown MAT-file version '{version}' specified")

    # if subsys_offset is not None:
    #     # For v7 version
    #     if subsys_offset > 0:
    #         write_subsystem_offset(file_stream, subsys_offset)

    #     with open(file_path, "wb") as f:
    #         f.write(file_stream.getvalue())

    # else:
    #     # For v7.3 version
    #     with open(file_path, "wb") as f:
    #         f.write(file_stream.getvalue())
    #         f.write(h5buf.getvalue())
