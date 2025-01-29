export PREFIX=/home/junjiey/work/fftisdf-benchmark/

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
    cmd=$(python $PREFIX/src/dump.py --prefix=$PREFIX --cell=$cell.vasp --kmesh=$kmesh --basis=$basis --method=$method)
    echo -e "$cmd" >> run.sh; sbatch --job-name=work/$cell/$basis/$method/$kmesh/ --cpus-per-task=64 --mem=480GB --time=08:00:00 run.sh --constraint=icelake

    # echo $cmd
    cd -
}


for cell in diamond-prim nio-prim diamond-conv nio-conv; do
    for basis in gth-szv-molopt-sr gth-dzvp-molopt-sr; do
        for kmesh in 1-1-2 1-2-2 2-2-2 2-2-4 2-4-4 4-4-4; do
            method=gdf
            submit $cell $kmesh $basis $method

            ke_cutoff=40
            method=fftdf
            submit $cell $kmesh $basis $method-$ke_cutoff

            for c0 in 10 20 30 40; do
                method=fftisdf-yang-$c0
                submit $cell $kmesh $basis $method-$ke_cutoff
            done

            ke_cutoff=80
            method=fftdf
            submit $cell $kmesh $basis $method-$ke_cutoff

            for c0 in 10 20 30 40; do
                method=fftisdf-yang-$c0
                submit $cell $kmesh $basis $method-$ke_cutoff
            done

            ke_cutoff=120
            method=fftdf
            submit $cell $kmesh $basis $method-$ke_cutoff

            for c0 in 10 20 30 40; do
                method=fftisdf-yang-$c0
                submit $cell $kmesh $basis $method-$ke_cutoff
            done

        done
    done
done

