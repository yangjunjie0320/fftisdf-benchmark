export PREFIX=/home/junjiey/work/fftisdf-benchmark/

function submit() {
    cell=$1
    kmesh=$2
    basis=$3
    ke_cutoff=$4
    method=$5

    dir=$PREFIX/work/$cell-$kmesh/$basis-$ke_cutoff/$method/
    printf "\n$dir\n"
    if [ -d $dir ]; then
        echo "Directory $dir already exists"
        rm -rf $dir
    fi

    mkdir -p $dir; cd $dir

    cp $PREFIX/src/run.sh run.sh
    cmd=$(python $PREFIX/src/dump.py --prefix=$PREFIX --cell=$cell.vasp --kmesh=$kmesh --ke_cutoff=$ke_cutoff --basis=$basis --method=$method)
    echo -e "$cmd" >> run.sh; sbatch --job-name=$method-$cell-$kmesh-$basis --cpus-per-task=64 --mem=480GB --time=48:00:00 run.sh

    # echo $cmd
    cd -
}

ke_cutoff=200
for cell in diamond-conv nio-conv cco-2x2-frac; do
    for basis in gth-szv-molopt-sr gth-dzvp-molopt-sr; do
        for kmesh in 1-1-2 1-2-2 2-2-2 2-2-4 2-4-4 4-4-4; do
            method=gdf
            # submit $cell $kmesh $basis $ke_cutoff $method

            for c0 in 10 20; do
                # method=fftisdf-yang-$c0-15-15-15
                # submit $cell $kmesh $basis $ke_cutoff $method

                method=fftisdf-ning-supercell-$c0
                submit $cell $kmesh $basis $ke_cutoff $method
            done

        done
    done
done

