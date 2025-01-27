for f in $(find . -name "main.py"); do
    echo "Found: $f"
    dir=$(dirname "$f")

    cd $dir; cp ../run.sh .; rm slurm-*

    sbatch \
      --job-name=$(basename $dir) \
      --cpus-per-task=64 \
      --mem=480GB \
      --time=48:00:00 \
      --constraint=icelake \
      ./run.sh
    
    cd -
done