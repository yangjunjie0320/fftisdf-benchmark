import numpy, scipy, os, sys
from pyscf.lib import chkfile

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="/home/junjiey/work/fftisdf-benchmark/work/nio-prim-2-2-2/gth-dzvp-molopt-sr-600/")
    args = parser.parse_args()

    # list all the directories in the prefix
    dirs = [os.path.join(args.prefix, d) for d in os.listdir(args.prefix) if os.path.isdir(os.path.join(args.prefix, d))]

    chk_ref = None
    if os.path.exists(os.path.join(args.prefix, "fftdf", "vjk.chk")):
        chk_ref = os.path.join(args.prefix, "fftdf", "vjk.chk")
        print("Using FFT-DF as reference.")
    elif os.path.exists(os.path.join(args.prefix, "gdf", "vjk.chk")):
        chk_ref = os.path.join(args.prefix, "gdf", "vjk.chk")
        print("Using GDF as reference.")
    else:
        raise ValueError("Reference file not found: %s" % chk_ref)

    vj_ref = chkfile.load(chk_ref, "vj")
    vk_ref = chkfile.load(chk_ref, "vk")

    info = {}

    for d in dirs:
        chk_sol = os.path.join(d, "vjk.chk")
        if not os.path.exists(chk_sol):
            continue

        info[os.path.basename(d)] = ""

        vj_sol = chkfile.load(chk_sol, "vj")
        vk_sol = chkfile.load(chk_sol, "vk")

        err_vj = abs(vj_ref - vj_sol).max()
        err_vk = abs(vk_ref - vk_sol).max()

        # get the time for the calculation
        with open(os.path.join(d, "out.log"), "r") as f:
            lines = f.readlines()
            t1 = float(lines[0].split()[-2])
            t2 = float(lines[1].split()[-2])
            t3 = float(lines[2].split()[-2])
            size = float(lines[3].split()[-2])

            # info[os.path.basename(d)] += "\n%s:" % os.path.basename(d)
            info[os.path.basename(d)] += "err_vj: %6.2e, " % err_vj
            info[os.path.basename(d)] += "err_vk: %6.2e, " % err_vk
            info[os.path.basename(d)] += "build:  %6.2e s, " % t1
            info[os.path.basename(d)] += "get_j:  %6.2e s, " % t2
            info[os.path.basename(d)] += "get_k:  %6.2e s, " % t3
            info[os.path.basename(d)] += "size:   %6.2e GB" % (size)

    for k, v in sorted(info.items()):
        print("%s # %s" % (v, k))
