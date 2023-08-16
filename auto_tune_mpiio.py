################## This is to infer or predict the MPI-IO bandwidth
################## on given set of parameters from the commandline
################## and auto-tune to new values with max bandwidth.

import sys
import torch

def main(argv):
    if len(argv) < 10:
        print("Usage: test_mpiio_model.py [READ/WRITE] \
              [chunk size] [NON_COLLECTIVE/COLLECTIVE] \
              [file size] [stripe count] [stripe size] \
              [number of mpi nodes] [processes per node] \
              [model filename]")
        quit()

    operation = argv[1]
    model_filename = argv[9]

    # Load the model
    model = torch.load(model_filename, map_location=torch.device('cpu'))
    model.eval()

    # Load the max values of dataset parameters in maxs.txt file.
    f = open("MPI-IO"+argv[1]+"_maxs.txt","r")

    max_stripe_size = float(f.readline())
    max_io_bandwidth = float(f.readline())
    max_mpi_nodes = float(f.readline())
    max_processes = float(f.readline())
    max_stripe_count = float(f.readline())
    max_chunk_size = float(f.readline())
    max_file_size = float(f.readline())
    f.close()
    
    # # Normalizing or scaling the parameters using MaxAbsScaler approach.
    collectivity = 0.0
    if argv[3] == "COLLECTIVE":
        collectivity = 1.0

    x = torch.tensor([[float(argv[2])/max_chunk_size, \
        collectivity, \
        float(argv[4])/max_file_size, \
        float(argv[5])/max_stripe_count, \
        float(argv[6])/max_stripe_size, \
        float(argv[7])/max_mpi_nodes, \
        float(argv[8])/max_processes \
    ]])
    x = x.view(7)
    current_max_bandwidth = model(x) * max_io_bandwidth    

    # Auto-tuning the parameters for MPI-IO READ operation.
    if operation == "READ":
        chunk_sizes = [1073741824,2147483648,4294967296,8589934592]
        access_types = ["NON_COLLECTIVE","COLLECTIVE"]
        max_prediction = current_max_bandwidth.item() 
        max_pred_chunk_size = float(argv[2])
        max_pred_access_type = argv[3]
        for curr_chunk_size in chunk_sizes:
            for curr_access_type in access_types:
                config_row = torch.Tensor(x)
                config_row[0] = float(curr_chunk_size) / max_chunk_size
                config_row[1] = 1.0 if curr_access_type == "COLLECTIVE" else 0.0
                predict_bandwidth = model(config_row) * max_io_bandwidth
                if predict_bandwidth.item() > max_prediction:
                    max_prediction = predict_bandwidth.item()
                    max_pred_chunk_size = curr_chunk_size
                    max_pred_access_type = curr_access_type

        ret_string = str(max_pred_chunk_size)+" "+max_pred_access_type+" "+str(max_prediction)
        print(ret_string)
        return ret_string

    # Auto-tuning the parameters for MPI-IO WRITE operation.
    if operation == "WRITE":
        chunk_sizes = [1073741824,2147483648,4294967296,8589934592]
        access_types = ["NON_COLLECTIVE","COLLECTIVE"]
        stripe_counts = [2,4,8,16]
        stripe_sizes = [1,256,512,1024,2048]
        max_prediction = current_max_bandwidth.item() 
        max_pred_chunk_size = float(argv[2])
        max_pred_access_type = argv[3]
        max_pred_stripe_count = float(argv[5])
        max_pred_stripe_size = float(argv[6])
        for curr_chunk_size in chunk_sizes:
            for curr_access_type in access_types:
                for curr_stripe_count in stripe_counts:
                    for curr_stripe_size in stripe_sizes:
                        config_row = torch.Tensor(x)
                        config_row[0] = float(curr_chunk_size) / max_chunk_size
                        config_row[1] = 1.0 if curr_access_type == "COLLECTIVE" else 0.0
                        config_row[3] = float(curr_stripe_count) / max_stripe_count
                        config_row[4] = float(curr_stripe_size) / max_stripe_size
                        predict_bandwidth = model(config_row) * max_io_bandwidth
                        if predict_bandwidth.item() > max_prediction:
                            max_prediction = predict_bandwidth.item()
                            max_pred_chunk_size = curr_chunk_size
                            max_pred_access_type = curr_access_type
                            max_pred_stripe_count = curr_stripe_count
                            max_pred_stripe_size = curr_stripe_size

        ret_string = str(max_pred_chunk_size)+" "+max_pred_access_type+" "+str(max_pred_stripe_count)+" "+str(max_pred_stripe_size)+" "+str(max_prediction)
        print(ret_string)
        return ret_string
        


if __name__ == "__main__":
   main(sys.argv)
