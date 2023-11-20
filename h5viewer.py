import h5py

# Open the HDF5 file in read-only mode
with h5py.File('model.h5', 'r') as file:
    # List the groups and datasets in the HDF5 file
    print("Groups and Datasets in HDF5 file:")
    for name in file:
        print(name)

