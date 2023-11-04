from re import X
import torch
from torch_geometric.datasets import QM9
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
dataset = QM9(root="../PaiNN/PaiNN/data")

print(dataset[0].keys())
print("number of graphs:\t\t",len(dataset))
print("number of classes:\t\t",dataset.num_classes)
print("number of node features:\t",dataset.num_node_features)
print("number of edge features:\t",dataset.num_edge_features)

# fig = plt.figure()
 
r_cut = torch.linalg.norm(dataset[0]['pos'][1] - dataset[0]['pos'][0], 2) + 0.1
print(f"R_cut will be {r_cut}")

# # syntax for 3-D projection
ax = plt.axes(projection ='3d')
diff_origin = dataset[0]['pos'] - torch.broadcast_to(dataset[0]['pos'][0], (dataset[0]['pos'].shape[0], 3))
norm_mol = torch.linalg.norm(diff_origin, dim=1)[1:]
print(norm_mol)
norm_idx = 0
for i in range(len(dataset[0]['pos'])):
        x = dataset[0]['pos'][i, 0]
        y = dataset[0]['pos'][i, 1]
        z = dataset[0]['pos'][i, 2]
        ax.scatter(x, y , z, c='r') 
        print(norm_mol[norm_idx-1])
        if norm_idx != 0 and norm_mol[norm_idx - 1] <= r_cut:
                print(True)
                ax.plot([x, dataset[0]['pos'][0, 0]], [y, dataset[0]['pos'][0, 1]], [z, dataset[0]['pos'][0, 2]]) 
        norm_idx +=1

plt.show()


# Plotting the atom position distribution
# pos_x = torch.empty((10000))
# pos_y = torch.empty((10000))
# pos_z = torch.empty((10000))
# for i in range(10000):
#     pos_x[i] = dataset[i]['pos'][0,0]
#     pos_y[i] = dataset[i]['pos'][0,1]
#     pos_z[i] = dataset[i]['pos'][0,2]
# sns.displot(x=pos_x, y=pos_y, kind="kde", fill=True)
# plt.show()
