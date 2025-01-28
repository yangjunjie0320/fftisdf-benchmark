for f in $(find . -name "main.py"); do
    echo "Found: $f"
    dir=$(dirname "$f"); cd $dir;

    if ! [ -f "run.sh" ]; then
        echo "Copying run.sh to $dir"
        cp ../run.sh .

        sbatch \
        --job-name=$(basename $dir) \
        --cpus-per-task=64 \
        --mem=480GB \
        --time=48:00:00 \
        --constraint=icelake \
        ./run.sh
    fi

    cd -
done