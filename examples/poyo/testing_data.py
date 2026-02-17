import h5py
import torch 

mask = torch.tensor([[True, True], [True, False], ], )
target = torch.ones_like(mask)
t2 = target[mask]
print(target.shape, t2.shape)
exit()

def print_structure(name, obj):
    print(name)


# with h5py.File("D:/Pose/Neuro Code/data/NoveltySessInfoMatFiles/linear/Achilles_10252013_sessInfo.mat", "r") as f:
#     f.visititems(print_structure)

#     print(list(f.keys()))  # top-level groups/datasets


# with h5py.File("D:/Pose/Neuro Code/poyo-reference/hippocampus/hippocampus_single_achilles.h5", "r") as f:
#     f.visititems(print_structure)

#     print(list(f.keys()))  # top-level groups/datasets
#     ['cursor', 'domain', 'spikes', 'trials', 'units']



print()
print('Another Processed Hippo data')

with h5py.File("D:/Pose/Neuro Code/data/NoveltySessInfoMatFiles/linear_processed/hippo_processed/achilles_10252013_sessinfo.h5", "r") as f: # buddy_06272013_sessinfo.h5
    f.visititems(print_structure)
    print(f['spikes']['timestamps'])
    print(f['train_domain']['start'][:], f['train_domain']['end'][:])

with h5py.File("D:/Pose/Neuro Code/data/NoveltySessInfoMatFiles/linear_processed/hippo_processed/buddy_06272013_sessinfo.h5", "r") as f: # 
    # f.visititems(print_structure)
    print(f['spikes']['timestamps'])
    print(f['train_domain']['start'][:], f['train_domain']['end'][:])


# print()
# print('Processed Hippo data')

# with h5py.File("D:/Pose/Neuro Code/data/NoveltySessInfoMatFiles/linear_processed/hippo_processed/Achilles_10252013_sessInfo_20010520_linear_maze.h5", "r") as f:
#     f.visititems(print_structure)

# print('\nPerich dataset')
# with h5py.File("D:/Pose/Neuro Code/data/poyo_datasets_processed/perich_miller_population_2018/c_20131003_center_out_reaching.h5", "r") as f:
#     f.visititems(print_structure)

    # for k in f.keys():
    #     print(k, type(f[k]))
    
    # print()

    # for k in ['trials']: 
    #     for kk in f[k].keys():
    #         print(k, kk, type(f[k][kk]))
    #         print(f[k][kk][:].shape)
    #         print(f[k][kk][:])
    #         print()
    #     # for kk in ['train_mask', 'valid_mask']: 
    #     #     # for kkk in f[k][kk].keys(): 
    #     #     print(k, kk, f[k][kk][:])

    # print()

    # print(f['cursor']['acc'][:].shape)

    # # print(list(f.keys()))  # top-level groups/datasets
    # # ['cursor', 'domain', 'spikes', 'trials', 'units']
