export PREFIX=$(pwd)

function submit() {
    cell=$1
    kmesh=$2
    basis=$3
    method=$4

    dir=$PREFIX/work/$cell/$basis/$method/$kmesh/
    printf "\n$dir\n"
    if [ -d $dir ]; then
        echo "Directory $dir already exists"
        rm -rf $dir
    fi

    mkdir -p $dir; cd $dir

    cp $PREFIX/src/run.sh run.sh
    cmd=$(python $PREFIX/src/dump.py --prefix=$PREFIX --cell=$cell.vasp --kmesh=$kmesh --basis=gth-$basis-molopt-sr --method=krhf-$method)
    echo -e "$cmd" >> run.sh; sbatch --job-name=work/$cell/$basis/$method/$kmesh/ --cpus-per-task=64 --mem=480GB --time=04:00:00 run.sh --constraint=icelake

    # echo $cmd
    cd -
}


for cell in diamond-prim; do
    for basis in dzvp; do
        for kmesh in 2-2-2 4-4-4; do
            method=gdf
            submit $cell $kmesh $basis $method

            ke_cutoff=40
            method=fftdf
            submit $cell $kmesh $basis $method-$ke_cutoff

            for c0 in 5 10 15 20 25 30 35 40; do
                method=fftisdf
                submit $cell $kmesh $basis $method-$c0-$ke_cutoff
            done
        done
    done
done
