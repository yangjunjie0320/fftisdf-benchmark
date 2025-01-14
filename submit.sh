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
    echo -e "$cmd" >> run.sh; sbatch --job-name=$method run.sh

    # echo $cmd
    cd -
}

submit diamond-prim 2-2-2 gth-dzvp-molopt-sr 200 gdf
submit diamond-prim 2-2-2 gth-dzvp-molopt-sr 200 fftdf
submit diamond-prim 2-2-2 gth-dzvp-molopt-sr 200 fftisdf-yang-20-15-15-15
submit diamond-prim 2-2-2 gth-dzvp-molopt-sr 200 fftisdf-yang-40-15-15-15
submit diamond-prim 2-2-2 gth-dzvp-molopt-sr 200 fftisdf-yang-60-15-15-15

submit diamond-prim 4-4-4 gth-dzvp-molopt-sr 200 gdf
submit diamond-prim 4-4-4 gth-dzvp-molopt-sr 200 fftdf
submit diamond-prim 4-4-4 gth-dzvp-molopt-sr 200 fftisdf-yang-20-15-15-15
submit diamond-prim 4-4-4 gth-dzvp-molopt-sr 200 fftisdf-yang-40-15-15-15
submit diamond-prim 4-4-4 gth-dzvp-molopt-sr 200 fftisdf-yang-60-15-15-15

submit diamond-prim 6-6-6 gth-dzvp-molopt-sr 200 gdf
submit diamond-prim 6-6-6 gth-dzvp-molopt-sr 200 fftdf
submit diamond-prim 6-6-6 gth-dzvp-molopt-sr 200 fftisdf-yang-20-15-15-15
submit diamond-prim 6-6-6 gth-dzvp-molopt-sr 200 fftisdf-yang-40-15-15-15
submit diamond-prim 6-6-6 gth-dzvp-molopt-sr 200 fftisdf-yang-60-15-15-15
