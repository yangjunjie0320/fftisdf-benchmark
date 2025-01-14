import numpy, scipy, os, sys
from pyscf.lib import chkfile

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, default="fftdf")
    parser.add_argument("--prefix", type=str, default="/home/junjiey/work/fftisdf-benchmark/work/nio-prim-2-2-2/gth-dzvp-molopt-sr-600/")
    args = parser.parse_args()

    # list all the directories in the prefix
    dirs = [os.path.join(args.prefix, d) for d in os.listdir(args.prefix) if os.path.isdir(os.path.join(args.prefix, d))]

    chk_ref = os.path.join(args.prefix, args.ref, "vjk.chk")
    assert os.path.exists(chk_ref), "Reference file not found: %s" % chk_ref

    vj_ref = chkfile.load(chk_ref, "vj")
    vk_ref = chkfile.load(chk_ref, "vk")

    for d in dirs:
        chk_sol = os.path.join(d, "vjk.chk")
        assert os.path.exists(chk_sol), "Solution file not found: %s" % chk_sol

        vj_sol = chkfile.load(chk_sol, "vj")
        vk_sol = chkfile.load(chk_sol, "vk")

        err_vj = abs(vj_ref - vj_sol).max()
        err_vk = abs(vk_ref - vk_sol).max()
        print("%s:\nerr_vj = %6.2e, err_vk = %6.2e" % (os.path.basename(d), err_vj, err_vk))

        # get the time for the calculation
        with open(os.path.join(d, "out.log"), "r") as f:
            lines = f.readlines()
            t1 = float(lines[0].split()[-2])
            t2 = float(lines[1].split()[-2])
            size = float(lines[-1].split()[-2])
            print("Building time = %6.2e, JK time = %6.2e" % (t1, t2))
            print("chk file size = %6.2e GB\n" % (size))

