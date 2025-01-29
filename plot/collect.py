import numpy, scipy, os, sys
from pyscf.lib import chkfile

def collect(d):
    assert os.path.exists(d), d

    vj = None
    vk = None

    if os.path.exists(os.path.join(d, "vjk.chk")):
        vj = chkfile.load(os.path.join(d, "vjk.chk"), "vj")
        vk = chkfile.load(os.path.join(d, "vjk.chk"), "vk")

    # info = {}
    res = {}
    for d1 in [os.path.join(d, t) for t in os.listdir(d) if os.path.isdir(os.path.join(d, t))]:
        info = ""
        kmesh = [int(x) for x in os.path.basename(d1).split("-")]
        info += "%4d, " % numpy.prod(kmesh)

        with open(os.path.join(d1, "out.log"), "r") as f:
            lines = f.readlines()
            t1 = float(lines[0].split()[-2]) if len(lines) > 0 else numpy.nan
            t2 = float(lines[1].split()[-2]) if len(lines) > 1 else numpy.nan
            t3 = float(lines[2].split()[-2]) if len(lines) > 2 else numpy.nan
            size = float(lines[3].split()[-2]) if len(lines) > 3 else numpy.nan

            # info[os.path.basename(d1)] = "" + ("%8d" % int(t1)) + ", "
            info += "" + ("%6.2e" % t1 if not numpy.isnan(t1) else "     nan") + ", "
            info += "" + ("%6.2e" % t2 if not numpy.isnan(t2) else "     nan") + ", "
            info += "" + ("%6.2e" % t3 if not numpy.isnan(t3) else "     nan") + ", "
            info += "" + ("%6.2e" % (size) if not numpy.isnan(size) else "     nan")

        vj = numpy.nan
        vk = numpy.nan
        if os.path.exists(os.path.join(d1, "vjk.chk")):
            vj = chkfile.load(os.path.join(d1, "vjk.chk"), "vj")
            vk = chkfile.load(os.path.join(d1, "vjk.chk"), "vk")
        
        res[numpy.prod(kmesh)] = {"info": info, "vj": vj, "vk": vk}
    return res

def parse(d1, d2):
    f = open(os.path.join("./data/%s-%s.log" % (os.path.basename(d1), os.path.basename(d2))), "w")

    dd = {}
    for d3 in [os.path.join(d2, x) for x in os.listdir(d2)]:
        if not os.path.isdir(d3):
            continue
        
        res = collect(d3)
        dd[os.path.basename(d3)] = {k: v for k, v in res.items()}

    ke_max = 0.0
    dd_ref = None

    for method, info in dd.items():        
        if "fftdf" in method:
            ke = float(method.split("-")[1])
            if ke > ke_max:
                ke_max = ke
                dd_ref = info

    if dd_ref is None:
        dd_ref = dd.get("gdf")

    # assert dd_ref is not None

    vj_ref = None
    vk_ref = None

    for k, v in dd_ref.items():
        if vj_ref is None:
            vj_ref = {}

        if vk_ref is None:
            vk_ref = {}

        vj_ref[k] = v["vj"]
        vk_ref[k] = v["vk"]

    for method, info in dd.items():
        for k, v in info.items():
            vj_sol = v.get("vj", numpy.inf)
            vk_sol = v.get("vk", numpy.inf)

            vj_err = numpy.array(abs(vj_sol - vj_ref.get(k, numpy.inf))).max()
            vk_err = numpy.array(abs(vk_sol - vk_ref.get(k, numpy.inf))).max()

            if numpy.isnan(vj_err) or numpy.isnan(vk_err):
                v["info"] += ",      nan,      nan"
            else:
                v["info"] += ", %6.2e, %6.2e" % (vj_err, vk_err)


    for method, info in sorted(dd.items()):
        f.write("# %s\n" % method)
        title = "%8s, " * 7
        title = title % ("nk", "build", "vj_time", "vk_time", "size", "vj_err", "vk_err")
        f.write("# " + title[6:-2] + "\n")
        for k, v in sorted(info.items()):
            f.write("%s\n" % v["info"])
        f.write("\n")
    f.close()

if __name__ == "__main__":
    prefix = "/home/junjiey/work/fftisdf-benchmark/work/"

    if not os.path.exists("./data"):
        os.makedirs("./data")

    for d1 in [os.path.join(prefix, x) for x in os.listdir(prefix)]:
        if not os.path.isdir(d1):
            continue

        for d2 in [os.path.join(d1, x) for x in os.listdir(d1)]:
            if not os.path.isdir(d2):
                continue

            parse(d1, d2)
            
            # try:
                
            # except Exception as e:
            #     print(e)
            #     continue

