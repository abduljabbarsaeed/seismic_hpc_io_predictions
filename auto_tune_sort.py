################ This is to infer or predict the SEG-Y Sort bandwidth
################ from given set of parameters from the commandline
################ and auto-tune to new values with max bandwidth.

import torch
import sys

def main(argv):
    #print(argv)
    if len(argv) < 10:
        print("Usage: test_fileio_model.py \
              [READ_CONTIGUOUS/WRITE_CONTIGUOUS] \
              [stripe count] [stripe size] \
              [number of mpi nodes] \
              [processes per node] [traces] [samples] \
              [UNIFORM_REVERSE/UNIFORM/RANDOM] \
              [model_filename]")
        quit()
    
    read_or_write = "READ" if argv[1] == "READ_CONTIGUOUS" else "WRITE"
    model_filename = argv[9]

    # Load the model.
    model = torch.load(model_filename, map_location=torch.device('cpu'))
    model.eval()

    # Load the max values of dataset parameters from maxs.txt file.
    f = open("SEG-Y-SORT"+read_or_write+"_maxs_new.txt","r")

    max_stripe_size = float(f.readline())
    max_io_bandwidth = float(f.readline())
    max_mpi_nodes = float(f.readline())
    max_processes = float(f.readline())
    max_stripe_count = float(f.readline())
    max_traces = float(f.readline())
    max_samples = float(f.readline())
    f.close()

    # Normalizing or scaling the parameters using MaxAbsScaler approach.
    unsorted_order = 0.0
    if argv[8] == "UNIFORM":
        unsorted_order = 0.5
    elif argv[8] == "RANDOM":
        unsorted_order = 1.0

    x = torch.tensor([[float(argv[2])/max_stripe_count, \
        float(argv[3])/max_stripe_size, \
        float(argv[4])/max_mpi_nodes, \
        float(argv[5])/max_processes, \
        float(argv[6])/max_traces, \
        float(argv[7])/max_samples, \
        unsorted_order \
    ]])
   
    x = x.view(7)
    current_max_bandwidth = model(x) * max_io_bandwidth 

    # Auto-tuning the parameters for SEG-Y sorting by either
    # READ_CONTIGUOUS or WRITE_CONTIGUOUS operation.  
    stripe_counts = [2,4,8,16]
    stripe_sizes = [1,256,512,1024,2048]
    max_prediction = current_max_bandwidth.item()
    max_pred_stripe_count = float(argv[2])
    max_pred_stripe_size = float(argv[3])
    for curr_stripe_count in stripe_counts:
        for curr_stripe_size in stripe_sizes:
            config_row = torch.Tensor(x)
            config_row[0] = float(curr_stripe_count) / max_stripe_count
            config_row[1] = float(curr_stripe_size) / max_stripe_size
            predict_bandwidth = model(config_row) * max_io_bandwidth
            predicted_bandwid = predict_bandwidth.item()
            if predicted_bandwid > max_prediction:
                max_prediction = predicted_bandwid
                max_pred_stripe_count = curr_stripe_count
                max_pred_stripe_size = curr_stripe_size

    ret_string = str(max_pred_stripe_count)+" "+str(max_pred_stripe_size)+" "+str(max_prediction)
    print(ret_string)
    return ret_string


if __name__ == "__main__":
   main(sys.argv)
