if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--basis", type=str, default="gth-dzvp-molopt-sr")
    parser.add_argument("--cell", type=str, default="nio-prim.vasp")
    parser.add_argument("--kmesh", type=str, default="2-2-2")
    # parser.add_argument("--ke_cutoff", type=float, default=200)
    parser.add_argument("--method", type=str, default=None)
    args = parser.parse_args()

    method = args.method
    assert method is not None

    prefix = args.prefix
    assert os.path.exists(prefix)

    cmd = "\n"
    if "fftisdf-yang" in method:
        assert os.path.exists("%s/src/main-%s.py" % (args.prefix, "fftisdf-yang")), "%s/src/main-%s.py" % (args.prefix, "fftisdf-yang")
        method = method.split("-")

        cmd += "cp %s/src/main-%s.py main.py\n" % (args.prefix, "-".join(method[:2]))
        cmd += "python main.py "
        cmd += "--c0=%.2f --ke_cutoff=%.2f " % (float(method[2]), float(method[3]))

    elif "fftdf" in method:
        assert os.path.exists("%s/src/main-%s.py" % (args.prefix, "fftdf"))
        cmd += "cp %s/src/main-%s.py main.py\n" % (args.prefix, method)
        cmd += "python main.py "
        cmd += "--ke_cutoff=%.2f " % float(method[1])

    elif "gdf" in method:
        assert os.path.exists("%s/src/main-%s.py" % (args.prefix, "gdf"))
        cmd += "cp %s/src/main-%s.py main.py\n" % (args.prefix, "gdf")
        cmd += "python main.py "

    elif "fftisdf-ning" in method:
        # is now abandoned
        raise RuntimeError("fftisdf-ning is now abandoned")
        method = method.split("-")
        assert os.path.exists("%s/src/main-%s.py" % (args.prefix, "-".join(method[:-1]))), "%s/src/main-%s.py" % (args.prefix, "-".join(method[:-1]))

        cmd += "export PYSCF_EXT_PATH=$HOME/packages/pyscf-forge/pyscf-forge-yangjunjie-non-orth/\n"
        cmd += "cp %s/src/main-%s.py main.py\n" % (args.prefix, "-".join(method[:-1]))
        c0 = float(method[-1])
        cmd += "python main.py "
        cmd += "--c0=%.2f " % c0

    cmd += "--cell=%s --kmesh=%s --basis=%s --pseudo=gth-pade" % (args.cell, args.kmesh, args.basis)
    cmd += "\n"

    print(cmd)
