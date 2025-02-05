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
    if "fftisdf" in method:
        c0 = method.split("-")[-2]
        k0 = method.split("-")[-1]
        method = method.split("-")[:-2]
        method = "-".join(method)
        main_path = "%s/src/main-%s.py" % (args.prefix, method)
        assert os.path.exists(main_path), "%s does not exist" % main_path

        cmd += "cp %s main.py\n" % main_path
        cmd += "python main.py "
        cmd += "--c0=%.2f --ke_cutoff=%.2f " % (float(c0), float(k0))

    elif "fftdf" in method:
        k0 = method.split("-")[-1]
        method = method.split("-")[:-1]
        method = "-".join(method)
        main_path = "%s/src/main-%s.py" % (args.prefix, method)
        assert os.path.exists(main_path), "%s does not exist" % main_path

        cmd += "cp %s main.py\n" % main_path
        cmd += "python main.py "
        cmd += "--ke_cutoff=%.2f " % float(k0)

    elif "gdf" in method:
        method = method.split("-")
        method = "-".join(method)
        main_path = "%s/src/main-%s.py" % (args.prefix, method)
        assert os.path.exists(main_path), "%s does not exist" % main_path

        cmd += "cp %s main.py\n" % main_path
        cmd += "python main.py "

    else:
        raise NotImplementedError

    cmd += "--cell=%s --kmesh=%s --basis=%s --pseudo=gth-pade" % (args.cell, args.kmesh, args.basis)
    cmd += "\n"

    print(cmd)
