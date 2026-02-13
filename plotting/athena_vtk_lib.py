
#!/usr/bin/env python
# coding: utf-8

"""Library for reading athenak vtk file. Adapted from the function 'vtk' in the
native athena script, 'athena/athena_read.py'.

Note that athenak vtk grid type is STRUCTURED_POINTS, while the athena vtk grid
type is RECTILINEAR_GRID.
"""

import re
import numpy as np
import struct


class AthenaError(RuntimeError):
    """General exception class for Athena++ read functions."""

    pass


def parse_time_from_header(file_path):
    """Parse for time from Athenak vtk file header.

    An example is:
    # vtk DataFile Version 2.0
    # Athena++ data at time= 50.2661  level= 0  nranks= 1  cycle=51640  variables=mhd_bcc

    Here, the time to be parsed is 50.2661.

    See https://gitlab.com/theias/hpc/jmstone/athena-parthenon/athenak/-/blame/master/src/outputs/vtk_mesh.cpp

    Args:
    file_path: str. Athena++ vtk file name.

    Returns:
    time: float or None. Parsed time or None.
    """

    time_pattern = re.compile(
        rb"time=\s*([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s*"
    )

    with open(file_path, "rb") as file:
        while True:
            line = file.readline()
            if not line:
                break  # End of file
            try:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("#"):
                    match = time_pattern.search(line)
                    if match:
                        time_value = float(match.group(1).decode("utf-8"))
                        return time_value
            except UnicodeDecodeError:
                # Ignore lines that cannot be decoded into text
                continue

    return None


def read_athenak_vtk(filename, array_names=None, squeeze=True, verbose=True):
    """Library for reading athenak vtk file. Adapted from the function 'vtk' in the
    native athena script, 'athena/athena_read.py'.

    Note that athenak vtk grid type is STRUCTURED_POINTS, while the athena vtk grid
    type is RECTILINEAR_GRID.

    Args:
    filename:
    array_names: list of array names, or athenak variable names to be retrieved;
        if empty, then all available variables will be saved.
    squeeze:
    verbose:

    Returns:
    time: float
    grid: dict with keys nxyz, xyzl, dxyz
    data: dict with keys of arrays
    """

    time = parse_time_from_header(filename)
    if verbose:
        print(f"time = {time}")

    #####################################
    # full, raw data (binary and ascii) #
    #####################################
    with open(filename, "rb") as data_file:
        raw_data = data_file.read()

    ##################################
    # decoded ascii data for parsing #
    ##################################
    raw_data_ascii = raw_data.decode("ascii", "replace")
    current_index = 0  # line number
    current_char = raw_data_ascii[current_index]

    #################
    # skip comments #
    #################
    while current_char == "#":
        while current_char != "\n":
            current_index += 1
            current_char = raw_data_ascii[current_index]
        current_index += 1
        current_char = raw_data_ascii[current_index]

    def skip_string(expected_string):
        """Skipping a string in the ascii data.

        This function will use temporary variables within the read_athena_vtk file
        context, such as raw_data_ascii and current_index.
        """
        expected_string_len = len(expected_string)
        if (
            raw_data_ascii[current_index : current_index + expected_string_len]
            != expected_string
        ):
            raise AthenaError("File not formatted as expected")
        return current_index + expected_string_len

    ################
    # parse header #
    ################
    current_index = skip_string(
        "BINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS "
    )
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != "\n":
        end_of_line_index += 1
    data_to_map = raw_data_ascii[current_index:end_of_line_index]
    face_dimensions = list(map(int, data_to_map.split(" ")))  # order: xyz
    current_index = end_of_line_index
    if verbose:
        print(f"face_dimensions = {face_dimensions}")

    current_index = skip_string("\nORIGIN ")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != "\n":
        end_of_line_index += 1
    data_to_map = raw_data_ascii[current_index:end_of_line_index]
    origin = list(map(float, data_to_map.split(" ")[:-1]))  # order: xyz
    current_index = end_of_line_index
    if verbose:
        print(f"origin = {origin}")

    current_index = skip_string("\nSPACING ")
    end_of_line_index = current_index + 1
    while raw_data_ascii[end_of_line_index] != "\n":
        end_of_line_index += 1
    data_to_map = raw_data_ascii[current_index:end_of_line_index]
    spacing = list(map(float, data_to_map.split(" ")[:-1]))  # order: xyz
    current_index = end_of_line_index
    if verbose:
        print(f"spacing = {spacing}")

    # Prepare to read quantities defined on grid
    cell_dimensions = np.array([max(dim - 1, 1) for dim in face_dimensions])
    num_cells = cell_dimensions.prod()
    if verbose:
        print(f"cell_dimensions = {cell_dimensions}")

    # num_cells = (face_dimensions[0]-1)*(face_dimensions[1]-1)
    current_index = skip_string("\n\nCELL_DATA {0}\n".format(num_cells))
    if raw_data_ascii[current_index : current_index + 1] == "\n":
        # extra newline inserted by join script
        current_index = skip_string("\n")
    data = {}

    def read_cell_scalars():
        begin_index = skip_string("SCALARS ")
        end_of_word_index = begin_index + 1
        while raw_data_ascii[end_of_word_index] != " ":
            end_of_word_index += 1
        array_name = raw_data_ascii[begin_index:end_of_word_index]
        string_to_skip = "SCALARS {0} float\nLOOKUP_TABLE default\n".format(
            array_name
        )
        begin_index = skip_string(string_to_skip)
        format_string = ">" + "f" * num_cells
        end_index = begin_index + 4 * num_cells

        if len(array_names) > 0 and array_name not in array_names:
            return end_index + 1

        data[array_name] = struct.unpack(
            format_string, raw_data[begin_index:end_index]
        )
        dimensions = tuple(cell_dimensions[::-1])
        data[array_name] = np.array(data[array_name]).reshape(dimensions)
        if squeeze:
            data[array_name] = data[array_name].squeeze()
        return end_index + 1

    while current_index < len(raw_data):
        expected_string = "SCALARS"
        expected_string_len = len(expected_string)
        if (
            raw_data_ascii[current_index : current_index + expected_string_len]
            == expected_string
        ):
            current_index = read_cell_scalars()
            continue
        else:
            break

    if verbose:
        print(data.keys())
    grid = dict(
        nxyz=cell_dimensions,
        xyzl=origin,
        dxyz=spacing,
    )

    return time, grid, data

