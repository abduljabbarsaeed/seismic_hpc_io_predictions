import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

np.random.seed(0)
torch.manual_seed(3)    # reproducible

#################### READING COMMAND LINE PARAMETERS ##############

if len(sys.argv)-1 < 5:
    print("usage: python3 <program_name>.py <benchmark> <operation> <data file path> <number of nodes in hidden layer 1> <number of nodes in hidden layer 2>")
    quit()

benchmark_name = sys.argv[1]
operation = sys.argv[2]
data_file_path = sys.argv[3]
H1 = sys.argv[4]
H2 = sys.argv[5]

#################### LOADING DATA #####################

mpio_read_data = pd.read_csv(data_file_path)

#################### SCALING OR NORMALIZTION ######################

f = open(benchmark_name+operation+"_maxs.txt","w+")

read_max_stripe_size = mpio_read_data["lustre_stripe_size"].max()
f.write(str(mpio_read_data["lustre_stripe_size"].max())+"\n")
mpio_read_data["lustre_stripe_size"] = mpio_read_data["lustre_stripe_size"] / \
                              mpio_read_data["lustre_stripe_size"].max()

read_max_io_bandwidth = mpio_read_data["io_bandwidth"].max()
f.write(str(mpio_read_data["io_bandwidth"].max())+"\n")
mpio_read_data["io_bandwidth"] = mpio_read_data["io_bandwidth"] / \
                                 mpio_read_data["io_bandwidth"].max()

read_max_mpi_nodes = mpio_read_data["mpi_number_of_nodes"].max() 
f.write(str(mpio_read_data["mpi_number_of_nodes"].max())+"\n")
mpio_read_data["mpi_number_of_nodes"] = mpio_read_data["mpi_number_of_nodes"] / \
                                 mpio_read_data["mpi_number_of_nodes"].max()

read_max_processes = mpio_read_data["mpi_processes_per_node"].max()
f.write(str(mpio_read_data["mpi_processes_per_node"].max())+"\n")
mpio_read_data["mpi_processes_per_node"] = mpio_read_data["mpi_processes_per_node"] / \
                                 mpio_read_data["mpi_processes_per_node"].max()

read_max_stripe_count = mpio_read_data["lustre_stripe_count"].max()
f.write(str(mpio_read_data["lustre_stripe_count"].max())+"\n")
mpio_read_data["lustre_stripe_count"] = mpio_read_data["lustre_stripe_count"] / \
                              mpio_read_data["lustre_stripe_count"].max()

##################### NORMALIZING MPI-IO SPECIFIC PARAMETERS ################################

if benchmark_name == "MPI-IO":
    mpio_read_data = mpio_read_data.drop(['uid','from_uid', \
                                      'benchmark','file_name_1', \
                                      'file_name_2','file_name_3', \
                                      'file_name_4','io_operation', \
                                       'max_nodes','mpi_backend'],axis=1)

    mpio_read_data["collectivity"].replace({"NON_COLLECTIVE":0.0, \
                                            "COLLECTIVE":1.0}, \
                                            inplace=True)

    mpio_read_data.astype({'chunk_size':'float32', \
                           'file_size':'float32', \
                           'lustre_stripe_size':'float32', \
                           'lustre_stripe_count':'float32', \
                           'mpi_number_of_nodes':'float32', \
                           'mpi_processes_per_node':'float32'}).dtypes


    read_max_chunk_size = mpio_read_data["chunk_size"].max()
    f.write(str(mpio_read_data["chunk_size"].max())+"\n")
    mpio_read_data["chunk_size"] = mpio_read_data["chunk_size"] / \
                                   mpio_read_data["chunk_size"].max()
    read_max_file_size = mpio_read_data["file_size"].max()
    f.write(str(mpio_read_data["file_size"].max())+"\n")
    mpio_read_data["file_size"] = mpio_read_data["file_size"] / \
                                  mpio_read_data["file_size"].max()

############################# NORMALIZING SEG-Y IO SPECIFIC PARAMETERS ################

if benchmark_name == "SEG-Y-IO":
    mpio_read_data = mpio_read_data.drop(['uid','from_uid', \
                                      'benchmark','file_name_1', \
                                      'file_name_2','file_name_3', \
                                      'file_name_4','io_direction', \
                                      'max_memory_usage','max_nodes', \
                                      'mpi_backend','random_seed'],axis=1)

    mpio_read_data["io_type"].replace({"CONTIGUOUS":0.0, \
                                       "RANDOM":1.0}, \
                                       inplace=True)

    mpio_read_data.astype({'number_of_traces':'float32', \
                           'samples_per_trace':'float32', \
                           'lustre_stripe_size':'float32',\
                           'lustre_stripe_count':'float32',\
                           'mpi_number_of_nodes':'float32',\
                           'mpi_processes_per_node':'float32'}).dtypes

    read_max_traces = mpio_read_data["number_of_traces"].max()
    f.write(str(read_max_traces)+"\n")
    mpio_read_data["number_of_traces"] = mpio_read_data["number_of_traces"] / \
                                  mpio_read_data["number_of_traces"].max()
    read_max_samples = mpio_read_data["samples_per_trace"].max()
    f.write(str(read_max_samples)+"\n")
    mpio_read_data["samples_per_trace"] = mpio_read_data["samples_per_trace"] / \
                                  mpio_read_data["samples_per_trace"].max() 

########################### NORMALIZING SEG-Y SORTING SPECIFIC PARAMETERS ##################

if benchmark_name == "SEG-Y-SORT":
    mpio_read_data = mpio_read_data.drop(['uid','from_uid', \
                                      'benchmark','file_name_1', \
                                      'file_name_2','file_name_3', \
                                      'file_name_4','contiguous_direction', \
                                      'max_memory_usage','max_nodes', \
                                      'mpi_backend'],axis=1)
    mpio_read_data["unsorted_order"].replace({"UNIFORM_REVERSE":0.0, \
                                       "UNIFORM":0.5, \
                                       "RANDOM":1.0}, \
                                       inplace=True)

    mpio_read_data.astype({'number_of_traces':'float32', \
                           'samples_per_trace':'float32', \
                           'lustre_stripe_size':'float32',\
                           'lustre_stripe_count':'float32',\
                           'mpi_number_of_nodes':'float32',\
                           'mpi_processes_per_node':'float32'}).dtypes

    read_max_traces = mpio_read_data["number_of_traces"].max()
    f.write(str(read_max_traces)+"\n")
    mpio_read_data["number_of_traces"] = mpio_read_data["number_of_traces"] / \
                                  mpio_read_data["number_of_traces"].max()
    read_max_samples = mpio_read_data["samples_per_trace"].max()
    f.write(str(read_max_samples)+"\n")
    mpio_read_data["samples_per_trace"] = mpio_read_data["samples_per_trace"] / \
                                  mpio_read_data["samples_per_trace"].max()

f.close()

################# DATA SAMPLING OR SHUFFLING #########################

mpio_read_data = mpio_read_data.sample(frac=1)
pd.set_option('display.max_columns', 8)
torch.set_printoptions(profile="full")
read_bandwidth_values = torch.Tensor(mpio_read_data["io_bandwidth"])

read_configs = torch.Tensor(mpio_read_data.drop(['io_bandwidth'],axis=1).values)
print(mpio_read_data.iloc[0:2,:])
print(read_configs[0:2,:])

################# DATA SPLITTING ################################

total_rows = int((80.0 / 100.0) * read_configs.size()[0]) # total rows are training rows
final_rows = int((20.0 / 100.0) * read_configs.size()[0]) # final rows are testing rows
total_cols = read_configs.size()[1]

test_csv = mpio_read_data.iloc[total_rows:total_rows+final_rows,:]
test_csv.to_csv(benchmark_name+"-"+operation+"-"+H1+"-"+H2+"-test-data.csv")

#quit()

######## For Cross-Validation if Require #####################

k = 1 

read_train_rmse_values = torch.empty(k,1)

read_test_rmse_values = torch.empty(k,1)

read_train_rr_value = torch.empty(k,1)
read_test_rr_value = torch.empty(k,1)

test_rows = int(total_rows / k) # partition size
train_rows = total_rows - test_rows
if k == 1:
    train_rows = total_rows
    test_rows = final_rows
#print(train_rows," ",test_rows," ",train_rows+test_rows)
#input()
X_train_read = 0
y_train_read = 0

X_pred_read = 0
y_pred_read = 0

curr_read_model = None

curr_read_mse = None
################################################################

################### INITIALIZING NEURAL NETWORK MODEL ###########

final_read_X = read_configs[total_rows:total_rows+final_rows,:]
final_read_y = read_bandwidth_values[total_rows:total_rows+final_rows]


n_in, n_h1, n_h2, n_out, batch_size = total_cols, int(H1), int(H2), 1, train_rows

max_iterations = {
                "MPI-IO_READ"      : 9570,
                "MPI-IO_WRITE"     : 5912,
                "SEG-Y-IO_READ"    : 27993,
                "SEG-Y-IO_WRITE"   : 3293,
                "SEG-Y-SORT_READ"  : 3941,
                "SEG-Y-SORT_WRITE" : 2726 
             }
print(max_iterations[benchmark_name+"_"+operation])                                            
hyper_parameters = {                        #lr   #de  #dp1 #dp2 #dp3
                      "MPI-IO_READ"      : [0.002,1e-5,0.00,0.00,0.00],
                      "MPI-IO_WRITE"     : [0.002,1e-5,0.05,0.05,0.05],
                      "SEG-Y-IO_READ"    : [0.002,0.00,0.00,0.00,0.00],
                      "SEG-Y-IO_WRITE"   : [0.002,0.00,0.00,0.05,0.05],
                      "SEG-Y-SORT_READ"  : [0.002,1e-5,0.00,0.05,0.05],
                      "SEG-Y-SORT_WRITE" : [0.002,0.00,0.00,0.00,0.00] 
                    }

for i in range(5):
    print(hyper_parameters[benchmark_name+"_"+operation][i])

model = nn.Sequential(nn.Dropout(p=hyper_parameters[benchmark_name+"_"+operation][2]),nn.ReLU(),nn.Linear(n_in, n_h1, bias=True), \
                      nn.Dropout(p=hyper_parameters[benchmark_name+"_"+operation][3]),nn.ReLU(), \
                      nn.Linear(n_h1, n_h2, bias=True), \
                      nn.Dropout(p=hyper_parameters[benchmark_name+"_"+operation][4]),nn.ReLU(), \
                      nn.Linear(n_h2, n_out, bias=True) \
                      #nn.Sigmoid() \
                    )
print(model)
for i in range (0,k):

    ################# FOR CROSS-VALIDATION IF REQUIRE ###################    
  
    start = i * test_rows
    X_train_read = torch.empty(train_rows,total_cols)

    y_train_read = torch.empty(train_rows,1)
    if k == 1:
        X_train_read = read_configs[0:train_rows,:]
        for j in range(0,train_rows):
            y_train_read[j][0] = read_bandwidth_values[j] 
    elif i == 0:
        X_train_read = read_configs[start+test_rows:total_rows,:]
        for j in range(0,train_rows):
            y_train_read[j][0] = read_bandwidth_values[j+test_rows] 
    elif i == (k - 1):
        X_train_read = read_configs[0:start,:]
        for j in range(0,train_rows):
            y_train_read[j][0] = read_bandwidth_values[j] 
    else:
        X_train_read = torch.cat((read_configs[0:start,:], \
                       read_configs[start+test_rows:total_rows,:]),0) 
        for j in range(0,start):
            y_train_read[j][0] = read_bandwidth_values[j]
        for j in range(start,train_rows):
            y_train_read[j][0] = read_bandwidth_values[j+test_rows] 
        print(X_train_read.size())
        print(y_train_read.size())
        quit()

    X_pred_read = torch.empty(test_rows,total_cols)
    X_pred_read = read_configs[start:start+test_rows,:]
    y_pred_read = torch.empty(test_rows,1)


    for j in range(0,test_rows):
        y_pred_read[j][0] = read_bandwidth_values[j+start]

    #################################################################

    ######################## TRAINING ON NN #####################

    if curr_read_model is not None:
        model = curr_read_model

    x = X_train_read

    y = y_train_read
    
    device = 0
    # if GPU is available, push tensors to it
    if torch.cuda.is_available():
        print("CUDA AVAILABLE!!")
        device = torch.device("cuda")
        x = x.to(device)
        y = y.to(device)
        model = model.to(device)
        #model_write = model_write.to(device)
        X_train_read  = X_train_read.to(device)
        y_train_read  = y_train_read.to(device)
        #X_train_write = X_train_read.to(device)
        #y_train_write = y_train_read.to(device)
        X_pred_read   = X_pred_read.to(device)
        y_pred_read   = y_pred_read.to(device)
        #X_pred_write  = X_pred_write.to(device)
        #y_pred_write  = y_pred_write.to(device)
        final_read_X = final_read_X.to(device)
        final_read_y = final_read_y.to(device)
    else:
        print("CUDA NOT AVAILABLE")



    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_parameters[benchmark_name+"_"+operation][0], \
                                 weight_decay=hyper_parameters[benchmark_name+"_"+operation][1])
    print(optimizer)
    
    y_train_read = y_train_read.view(train_rows)
    for epoch in range(max_iterations[benchmark_name+"_"+operation]):

        # Forward Propagation
        y_pred_train = model(x)
        # Compute and print loss
        #print(">>>>>>",y_pred_train.size()," ",y_train_read.size())
        loss = criterion(y_pred_train.view(train_rows), y_train_read)
        #if loss.item() <= 1.20e-06:
        #    break
        # Zero the gradients
        optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        optimizer.step()
        with torch.no_grad():
            model.eval()
            y_test = model(final_read_X)
            loss_test = criterion(y_test.view(final_rows), final_read_y)
            errors = torch.abs(final_read_y - y_test.view(final_rows))
            mape = torch.div(errors, final_read_y) * 100.0
            accuracy = 100.0 - torch.mean(mape)
        print('epoch: ', epoch,' loss: ', loss.item(), ' loss_test: ', loss_test.item(), 'accuracy: ', accuracy.item())
        
        model.train()

    read_train_rmse_values[i] = torch.mean((y_pred_train-y_train_read)**2)
    print(read_train_rmse_values[i])
    #input()
    Accuracy_train_read = 100 - torch.div(torch.abs((y_train_read - y_pred_train)), torch.max(y_train_read,y_pred_train)) * 100
    torch.set_printoptions(threshold=10000)
    mse_train_read = torch.div(torch.sum(torch.abs(y_train_read - y_pred_train)),train_rows)
    print("mse on training set = ", mse_train_read)
    

    print("Average accuracy for read on training set is ", torch.mean(torch.abs(Accuracy_train_read)))

    relative_residual_train_read = torch.div(torch.dist(y_train_read, y_pred_train, p=2),torch.norm(y_train_read, p=2))
    print("relative residual for read on training set is ", relative_residual_train_read)

    y_pred_new = torch.abs(model(X_pred_read))
    print(y_pred_new.min())
    read_test_rmse_values[i] = torch.mean((y_pred_new-y_pred_read)**2)
    print(read_test_rmse_values[i])
    #input()
    if curr_read_mse is None or read_test_rmse_values[i] < curr_read_mse:
        curr_read_mse = read_test_rmse_values[i]
        curr_read_model = model


    Accuracy_pred_read = 100 - torch.div(torch.abs((y_pred_read - y_pred_new)), torch.max(y_pred_read,y_pred_new)) * 100
    torch.set_printoptions(threshold=10000)

    print("Average accuracy for read on new set is ", torch.mean(torch.abs(Accuracy_pred_read)))

    relative_residual_pred_read = torch.div(torch.dist(y_pred_read, y_pred_new, p=2),torch.norm(y_pred_read, p=2))
    print("relative residual for read on new set is ", relative_residual_pred_read)
    mse_test_read = torch.div(torch.sum(torch.abs(y_pred_read - y_pred_new)), test_rows)
    print("mse on test set", mse_test_read)
    
################## COMPUTING PREDICTION RESULTS ################

final_read_pred = torch.abs(curr_read_model(final_read_X))
final_read_pred = final_read_pred.view(final_rows)


print(final_read_pred.max() * read_max_io_bandwidth)
print(final_read_pred.min() * read_max_io_bandwidth)

final_read_rmse = torch.mean((torch.abs(final_read_pred)-final_read_y)**2)

print(benchmark_name,' ',operation,' train avg rmse(s) ',read_train_rmse_values.mean())
print(benchmark_name,' ',operation,' test avg rmse(s) ',read_test_rmse_values.mean())
print('\n')
print(benchmark_name,' ',operation,' final rmse ', final_read_rmse)

################# SAVING THE MODEL ################
    
torch.save(curr_read_model, benchmark_name+"_"+operation+"_"+H1+"_"+H2+"_MODEL-new.pt")

################ COMPUTING ACCURACY RESULTS ON PREDICTED IO BANDWIDTHS ########

read_actual_sum = 0
read_actual_avg = 0
read_sum = 0
read_avg = 0
read_accu_temp = 0
read_actual_diff_sum = 0
read_actual_accu_temp = 0
read_temp_sum = 0
read_actual_temp_sum = 0
above_90 = 0
above_80 = 0
above_70 = 0
above_60 = 0
above_50 = 0
above_40 = 0
above_30 = 0
above_20 = 0
above_10 = 0
above_00 = 0
above_100 = 0
else_under=0

above_100_accu = 0
above_90_accu = 0
above_80_accu = 0
above_70_accu = 0
above_60_accu = 0
above_50_accu = 0
above_40_accu = 0
above_30_accu = 0
above_20_accu = 0
above_10_accu = 0
above_00_accu = 0
under_00_accu = 0

print(total_rows)
print(final_rows)
print(total_rows+final_rows)

for i in range(final_rows):

    read_actual_diff = abs(final_read_y[i] - final_read_pred[i]) * read_max_io_bandwidth
    read_actual_diff_sum = read_actual_diff_sum + read_actual_diff
    read_diff = abs(final_read_y[i] - final_read_pred[i])
    read_actual_max = max(final_read_y[i], final_read_pred[i]) * read_max_io_bandwidth
    read_max = max(final_read_y[i], final_read_pred[i])
    read_actual_perc = float(read_actual_diff) / (final_read_y[i] * read_max_io_bandwidth) * 100.0 #float(read_actual_max) * 100.0

    
    read_perc = float(read_diff) / final_read_y[i] * 100.0 #float(read_max) * 100.0
    read_actual_accu_temp = read_actual_perc
    read_accu_temp = read_perc
    read_actual_temp_sum = read_actual_temp_sum + read_actual_accu_temp
    read_temp_sum = read_temp_sum + read_accu_temp
    read_actual_accu = 100.0 - read_actual_perc
    if read_actual_accu >= 100.0:
        above_100 = above_100 + 1
        above_100_accu += read_actual_accu
    elif read_actual_accu >= 90.0:
        above_90 = above_90 + 1
        above_90_accu += read_actual_accu
    elif read_actual_accu >= 80.0:
        above_80 = above_80 + 1
        above_80_accu += read_actual_accu
    elif read_actual_accu >= 70.0:
        above_70 = above_70 + 1
        above_70_accu += read_actual_accu
    elif read_actual_accu >= 60.0:
        above_60 = above_60 + 1
        above_60_accu += read_actual_accu
    elif read_actual_accu >= 50.0:
        above_50 = above_50 + 1
        above_50_accu += read_actual_accu
    elif read_actual_accu >= 40.0:
        above_40 = above_40 + 1
        above_40_accu += read_actual_accu
    elif read_actual_accu >= 30.0:
        above_30 = above_30 + 1
        above_30_accu += read_actual_accu
    elif read_actual_accu >= 20.0:
        above_20 = above_20 + 1
        above_20_accu += read_actual_accu
    elif read_actual_accu >= 10.0:
        above_10 = above_10 + 1
        above_10_accu += read_actual_accu
    elif read_actual_accu >= 0.0:
        above_00 = above_00 + 1
        above_00_accu += read_actual_accu
    else:
        else_under=else_under+1
        under_00_accu += read_actual_accu
        '''print("read_actual_accu=",read_actual_accu)
        print("read_actual_perc=",read_actual_perc)
        print("read_actual_diff=",read_actual_diff)
        print("final_read_pred[",i,"]=",final_read_pred[i]*read_max_io_bandwidth)
        print("final_read_y[",i,"]=",final_read_y[i]*read_max_io_bandwidth)'''

    read_accu = 100.0 - read_perc
    read_actual_sum = read_actual_sum + read_actual_accu
    read_sum = read_sum + read_accu

print(curr_read_model)
errors = torch.abs(final_read_y - final_read_pred)
mape = torch.div(errors, final_read_y) * 100.0
accuracy = 100.0 - torch.mean(mape)
print("accuracy = ", accuracy,"%.")
print("read_actual temp = ", 100.0 - read_actual_temp_sum / float(final_rows),"%.")
print("read_ temp = ", 100.0 - read_temp_sum / float(final_rows),"%.")    
read_actual_avg = float(read_actual_sum) / float(final_rows)
read_avg = float(read_sum) / float(final_rows)
read_mae = float(read_actual_diff_sum) / float(final_rows)
print("read_actual_avg=",read_actual_avg, "%.")
print("read_avg=",read_avg,"%.")
print("read_mae=",read_mae)
print("iterations= ",epoch)
print("above_100 = ", above_100)
print("above_90 = ", above_90)
print("above_80 = ", above_80)
print("above_70 = ", above_70)
print("above_60 = ", above_60)
print("above_50 = ", above_50)
print("above_40 = ", above_40)
print("above_30 = ", above_30)
print("above_20 = ", above_20)
print("above_10 = ", above_10)
print("above_00 = ", above_00)
print("else_under = ", else_under)
temp_accu = 0
no_accu = 0 
if above_90 > 0:
    print("above_90_accu = ", above_90_accu/float(above_90),"%.")
    temp_accu += above_90_accu
    no_accu += above_90
if above_80 > 0:
    print("above_80_accu = ", above_80_accu/float(above_80),"%.")
    temp_accu += above_80_accu
    no_accu += above_80
if above_70 > 0:
    print("above_70_accu = ", above_70_accu/float(above_70),"%.")
    temp_accu += above_70_accu
    no_accu += above_70 
if above_60 > 0:
    print("above_60_accu = ", above_60_accu/float(above_60),"%.")
    temp_accu += above_60_accu
    no_accu += above_60
if above_50 > 0:
    print("above_50_accu = ", above_50_accu/float(above_50),"%.")
    temp_accu += above_50_accu
    no_accu += above_50
if above_40 > 0:
    print("above_40_accu = ", above_40_accu/float(above_40),"%.")
    temp_accu += above_40_accu
    no_accu += above_40
if above_30 > 0:
    print("above_30_accu = ", above_30_accu/float(above_30),"%.")
    temp_accu += above_30_accu
    no_accu += above_30
if above_20 > 0:
    print("above_20_accu = ", above_20_accu/float(above_20),"%.")
    temp_accu += above_20_accu
    no_accu += above_20
if above_10 > 0:
    print("above_10_accu = ", above_10_accu/float(above_10),"%.")
    temp_accu += above_10_accu
    no_accu += above_10
if above_00 > 0:
    print("above_00_accu = ", above_00_accu/float(above_00),"%.")
    temp_accu += above_00_accu
    no_accu += above_00
if else_under > 0:
    print("under_00_accu = ", under_00_accu/float(else_under),"%.")
    temp_accu += under_00_accu
    no_accu += else_under
print("accuracy excluding under 00 cases = ", 
     (temp_accu - under_00_accu) / float(no_accu - else_under),"%."
     )
print("accuracy including under 00 cases = ", 
     (temp_accu) / float(no_accu),"%."
     )
print("accuracy.item()=", accuracy.item(),"%.")
print("training mse = ", loss.item())
print("testing mse = ", loss_test.item())

# Generating Predictions Graph

X_all_read = [i+1 for i in range(50)]
y_all_final_read = (final_read_y[0:50] * read_max_io_bandwidth).tolist()
y_all_pred_read = (final_read_pred[0:50] * read_max_io_bandwidth).tolist()

plt.plot(X_all_read, y_all_final_read, \
           label="Actual values", \
           linestyle='solid', \
           marker = 'o', \
           markerfacecolor='red', \
           markersize=7
        )
plt.plot(X_all_read, y_all_pred_read, \
           label="Predictions", \
           linestyle='solid', \
           marker = 's', \
           markerfacecolor='green', \
           markersize=5
        )
plt.xlabel('Configurations')
# naming the y axis 
plt.ylabel('I/O Bandwidth (MB/s)')
# giving a title to my graph 
plt.title(benchmark_name+' '+operation+': H1='+H1+', H2='+H2) 
# show a legend on the plot 
plt.legend()
# function to show the plot 
#plt.show()
# save plot as figure .png image
plt.savefig(benchmark_name+'_'+operation+"_"+H1+"_"+H2+'_PREDICTIONS-new.png')
