# graphlink

Project based on High Performance Graph Analytics

*Cluster visualization and Clustering modularity scores are calculated for partitions created by parallel running k-means

*Applications are discussed and Performance is evaluated to compare with traditional k-means algorithms


Steps to run on env

1) srun -p gpu -A general --gpus-per-node 1 --pty bash
2) module avail deeplearning
3) module load deeplearning


Run main file
python main.py

Run visualization.ipynb file
jupyter nbconvert --to notebook --execute visualization.ipynb --output visualization.ipynb