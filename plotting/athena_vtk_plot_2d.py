############
# PREAMBLE #
############

import argparse
import pathlib
import sys

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from athena_vtk_lib import read_athenak_vtk

##################################
# PARSING COMMAND LINE ARGUMENTS #
##################################


def str2cmap(name):
    if name not in plt.colormaps():
        raise argparse.ArgumentTypeError(
            f"'{name}' is not a valid colormap name."
            f" Choose from: {', '.join(plt.colormaps())}"
        )
    return plt.get_cmap(name)



def main(args):

    ################
    # READING DATA #
    ################
    filename = args.filename
    fieldname = args.fieldname
    vscale = args.vscale
    cmap = args.cmap

    if fieldname is None:
        fieldnames = []
    else:
        fieldnames = [fieldname]

    my_time, grid, data = read_athenak_vtk(filename, fieldnames, verbose=False)

    dx, dy, _ = grid["dxyz"]
    dx, dy, _ = grid["dxyz"]
    nx, ny, _ = grid["nxyz"]
    xl, yl, _ = grid["xyzl"]

    xh = xl + dx * nx
    yh = yl + dy * ny

    ###############
    # MAKING PLOT #
    ###############

    # if there is only one field, automatically use the only available field
    if fieldname is None:
        if len(data.keys()) > 1:
            msg = (
                f"fieldname is not set but there are more than one field available"
            )
            msg += f"; available fields are {data.keys()}"
            raise ValueError(msg)
        else:
            fieldname = [k for k in data.keys()][0]


    if fieldname in data:



        field_data = data[fieldname]

        vmin, vmax = args.vmin, args.vmax
        _vmin, _vmax = field_data.min(), field_data.max()
        if vmin is None:
            vmin = _vmin
        if vmax is None:
            vmax = _vmax

        # find color mapping method
        if vscale == "linear":
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        elif vscale == "log":
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        elif vscale == "symmetric":
            halfrange = max(abs(vmin), abs(vmax))
            norm = mpl.colors.CenteredNorm(halfrange=None)
        else:
            raise ValueError(f"invalid --vscale option '{vscale}'")

        fig, ax = plt.subplots(figsize=args.figsize)

        im = ax.imshow(
            field_data,
            extent=(xl, xh, yl, yh),
            origin="lower",
            cmap=cmap,
            norm=norm,
        )

        fig.colorbar(im, ax=ax)

        title = args.title
        title = title.replace("__time__", f"{my_time:g}")
        title = title.replace("__fieldname__", f"{fieldname}")
        ax.set_title(title)
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(args.ylabel)

        if args.imagename is None:
            plt.show()
        else:
            plt.savefig(args.imagename, bbox_inches="tight", dpi=320)

    else:
        print("SKIP")

if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        description="""Make contour plots of using athenak 2d vtk file.""",
    )

    parser.add_argument("--vmin", default=None, type=float)
    parser.add_argument("--vmax", default=None, type=float)
    parser.add_argument(
        "--vscale",
        default="linear",
        choices=("linear", "log", "symmetric"),
    )
    parser.add_argument(
        "--cmap",
        type=str2cmap,
        default=None,
        help="Colormap for the time colorbar.",
    )
    parser.add_argument("--figsize", default=None, nargs=2, type=float)
    parser.add_argument("--title", default="")
    parser.add_argument("--xlabel", default="")
    parser.add_argument("--ylabel", default="")
    parser.add_argument("--imagename", default='alfredo.png')

    args = parser.parse_args()

    for name in ['mhd_bcc', 'mhd_jz', 'mhd_w']:
        for num in ['00000', '00001', '00002', '00003', '00004']:
            fname = f'HB3.{name}.{num}.vtk'
            path=f"/mnt/home/mgoldstein/ceph/athenak/vtk/{fname}"
            args.filename = path
            args.fieldname = 'velx'
            main(args=args)


