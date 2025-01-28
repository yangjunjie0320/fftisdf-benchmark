import numpy as np                                                                      
import h5py
from mpi4py import MPI 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main():
    print(f"Rank={rank} out of {size} total ranks.", flush=True)
    comm.Barrier()
    
    # Define the size of the (square) matrix
    matrix_size = 512  # Adjust as needed for a larger example
    
   # Open/create the file in parallel mode
    with h5py.File('matrix_data.h5', 'w', driver='mpio', comm=comm) as f:
        # Rank=0 creates the dataset; other ranks collectively participate
        dset = f.create_dataset(
            'big_matrix', 
            shape=(matrix_size, matrix_size), 
            dtype=np.float64
        )
        
        for i in range(matrix_size):
            if not i % size == rank:
                continue

            dset[i, :] += rank

    comm.Barrier()
    
    if rank == 0:
        print("Rank 0 has finished writing the dataset.", flush=True)

    norm_local = 0.0
    with h5py.File('matrix_data.h5', 'r', driver='mpio', comm=comm) as f:
        dset = f['big_matrix']

        for i in range(matrix_size):
            if not i % size == rank:
                continue

            norm_local += np.linalg.norm(dset[:, i])
            print(f"reading row {i} of {matrix_size} in rank {rank} / {size}, norm_local = {norm_local}", flush=True)
                  
    norm = comm.allreduce(norm_local, op=MPI.SUM)
    comm.Barrier()
    
    print(f"Rank {rank} has norm {norm}", flush=True)

if __name__ == "__main__":
    main()
