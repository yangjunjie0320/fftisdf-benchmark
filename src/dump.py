if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--basis", type=str, default="gth-dzvp-molopt-sr")
    parser.add_argument("--cell", type=str, default="nio-prim.vasp")
    parser.add_argument("--kmesh", type=str, default="2-2-2")
    parser.add_argument("--ke_cutoff", type=float, default=200)
    parser.add_argument("--method", type=str, default=None)
    args = parser.parse_args()

    method = args.method
    assert method is not None

    prefix = args.prefix
    assert os.path.exists(prefix)

    cmd = "\n"
    if "fftisdf-yang" in method:
        assert os.path.exists("%s/src/main-%s.py" % (args.prefix, "fftisdf-yang"))
        method = method.split("-")

        cmd += "cp %s/src/main-%s.py main.py\n" % (args.prefix, "fftisdf-yang")
        cmd += "python main.py "
        c0 = float(method[2])
        m0 = "-".join(method[3:])
        cmd += "--c0=%.2f --m0=%s " % (c0, m0)
    
    elif "fftisdf-ning" in method:
        assert os.path.exists("%s/src/main-%s.py" % (args.prefix, "fftisdf-ning"))
        method = method.split("-")

        cmd += "export PYSCF_EXT_PATH=$HOME/packages/pyscf-forge/pyscf-forge-yangjunjie-non-orth/\n"
        cmd += "cp %s/src/main-%s.py main.py\n" % (args.prefix, "fftisdf-ning")
        c0 = float(method[2])
        cmd += "python main.py "
        cmd += "--c0=%.2f " % c0

    else:
        assert os.path.exists("%s/src/main-%s.py" % (args.prefix, method))
        cmd += "cp %s/src/main-%s.py main.py\n" % (args.prefix, method)
        cmd += "python main.py "

    cmd += "--cell=%s --kmesh=%s --basis=%s " % (args.cell, args.kmesh, args.basis)
    cmd += "--ke_cutoff=%.2f --pseudo=gth-pade " % args.ke_cutoff
    cmd += "\n"

    print(cmd)
