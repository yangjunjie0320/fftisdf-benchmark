import numpy, scipy, os, sys
from pyscf.lib import chkfile

def collect(d, ref=None):
    assert os.path.exists(d), d

    # find the fftdf directory or the gdf directory
    fftdf_dir = None
    gdf_dir = None
    for t in os.listdir(d):
        if os.path.isdir(os.path.join(d, t)) and "fftdf" in t:
            fftdf_dir = os.path.join(d, t)
        elif os.path.isdir(os.path.join(d, t)) and "gdf" in t:
            gdf_dir = os.path.join(d, t)

    ref = fftdf_dir if fftdf_dir is not None else gdf_dir
    vj_ref = None
    vk_ref = None

    if ref is not None:
        from pyscf.lib import chkfile
        vj_ref = chkfile.load(os.path.join(ref, "vjk.chk"), "vj")
        vk_ref = chkfile.load(os.path.join(ref, "vk.chk"), "vk")

    info = {}
    for d1 in [os.path.join(d, t) for t in os.listdir(d) if os.path.isdir(os.path.join(d, t))]:
        info[os.path.basename(d1)] = ""

        with open(os.path.join(d1, "out.log"), "r") as f:
            lines = f.readlines()
            t1 = float(lines[0].split()[-2]) if len(lines) > 0 else numpy.nan
            t2 = float(lines[1].split()[-2]) if len(lines) > 1 else numpy.nan
            t3 = float(lines[2].split()[-2]) if len(lines) > 2 else numpy.nan
            size = float(lines[3].split()[-2]) if len(lines) > 3 else numpy.nan

            # info[os.path.basename(d1)] = "" + ("%8d" % int(t1)) + ", "
            info[os.path.basename(d1)] += "" + ("%6.2e" % t1 if not numpy.isnan(t1) else "     nan") + ", "
            info[os.path.basename(d1)] += "" + ("%6.2e" % t2 if not numpy.isnan(t2) else "     nan") + ", "
            info[os.path.basename(d1)] += "" + ("%6.2e" % t3 if not numpy.isnan(t3) else "     nan") + ", "
            info[os.path.basename(d1)] += "" + ("%6.2e" % (size) if not numpy.isnan(size) else "     nan")

        if vj_ref is not None:
            vj_sol = chkfile.load(os.path.join(d1, "vjk.chk"), "vj")
            info[os.path.basename(d1)] += ", %6.2e" % abs(vj_ref - vj_sol).max()
        else:
            info[os.path.basename(d1)] += ", nan"

        if vk_ref is not None:
            vk_sol = chkfile.load(os.path.join(d1, "vk.chk"), "vk")
            info[os.path.basename(d1)] += ", %6.2e" % abs(vk_ref - vk_sol).max()
        else:
            info[os.path.basename(d1)] += ", nan"

    res = []
    for k, v in sorted(info.items()):
        kmesh = [int(x) for x in k.split("-")]
        res.append("%4d, %s" % (numpy.prod(kmesh), v))
    return "\n".join(res)

if __name__ == "__main__":
    prefix = "/Users/yangjunjie/work/fftisdf-benchmark/work/"
    dirs = [os.path.join(prefix, d) for d in os.listdir(prefix) if os.path.isdir(os.path.join(prefix, d))]

    if not os.path.exists("./data"):
        os.makedirs("./data")

    for d1 in [os.path.join(prefix, x) for x in os.listdir(prefix)]:
        if not os.path.isdir(d1):
            continue

        for d2 in [os.path.join(d1, x) for x in os.listdir(d1)]:
            if not os.path.isdir(d2):
                continue

            f = open(os.path.join("./data/%s-%s.log" % (os.path.basename(d1), os.path.basename(d2))), "w")

            dd = {}
            for d3 in [os.path.join(d2, x) for x in os.listdir(d2)]:
                if not os.path.isdir(d3):
                    continue
                
                dd[os.path.basename(d3)] = "# %2s, %8s, %8s, %8s, %8s, %8s, %8s" % ("nk", "build", "get_j", "get_k", "size", "vj_err", "vk_err")
                dd[os.path.basename(d3)] += "\n%s" % collect(d3)
            
            f.write("\n\n".join("# %s\n%s" % (k, v) for k, v in sorted(dd.items())))

            f.close()
