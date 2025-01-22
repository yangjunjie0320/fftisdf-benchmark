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

ke_cutoff=20
for cell in diamond-conv nio-conv; do
    for basis in gth-szv-molopt-sr gth-dzvp-molopt-sr; do
        for kmesh in 2-2-2 4-4-4 6-6-6; do
            for method in fftdf gdf; do
                submit $cell $kmesh $basis $ke_cutoff $method
            done

            for c0 in 10 20 30 40; do
                method=fftisdf-yang-$c0-19-19-19
                submit $cell $kmesh $basis $ke_cutoff $method

                method=fftisdf-ning-$c0
                submit $cell $kmesh $basis $ke_cutoff $method
            done

        done
    done
done

# ke_cutoff=200
# for cell in nio-prim nio-conv; do
#     for basis in gth-szv-molopt-sr gth-dzvp-molopt-sr; do
#         for kmesh in 2-2-2 4-4-4 6-6-6; do
#             for method in fftdf gdf; do
#                 submit $cell $kmesh $basis $ke_cutoff $method
#             done

#             for c0 in 10 20 30 40; do
#                 method=fftisdf-yang-$c0-19-19-19
#                 submit $cell $kmesh $basis $ke_cutoff $method

#                 method=fftisdf-ning-$c0
#                 submit $cell $kmesh $basis $ke_cutoff $method
#             done
            
#         done
#     done
# done

# ke_cutoff=200
# for cell in cacuo2-afm; do
#     for basis in gth-szv-molopt-sr; do
#         for kmesh in 2-2-2 4-4-4; do
#             # for method in gdf; do
#             #     submit $cell $kmesh $basis $ke_cutoff $method
#             # done

#             for c0 in 10 20 30 40; do
#                 method=fftisdf-yang-$c0-19-19-19
#                 submit $cell $kmesh $basis $ke_cutoff $method

#                 # method=fftisdf-ning-$c0
#                 # submit $cell $kmesh $basis $ke_cutoff $method
#             done
#         done
#     done
# done
