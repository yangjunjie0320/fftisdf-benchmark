for f in $(find . -name "run.sh"); do
    echo "Found: $f"
    dir=$(dirname "$f")

    cd $dir; rm sl* out.log;
    
    sbatch \
      --job-name=diamond \
      --cpus-per-task=64 \
      --mem=480GB \
      --time=48:00:00 \
      --constraint=icelake \
      ./run.sh
    
    cd -
done