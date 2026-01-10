import torch, os
from torch_geometric.data import Data
import logging
import matplotlib.pyplot as plt
import numpy as np

from read_data import load_downstream_tracks, calc_avg_ut_xy_position

logging.basicConfig(level=logging.INFO, format='%(message)s')

plt.rcParams.update({
    "font.size":10,
    "axes.titlesize":12,
    "axes.labelsize":11,
    "legend.fontsize":10,
    "xtick.labelsize":10,
    "ytick.labelsize":10,})


def build_graph_from_track(track):
    #NODES
    node_ft=[]
    node_z=[]

    det_type={"UT":0, "SciFi":1}
    z0=track["ut_z_position"][0]
    y0=track["y_position"]

    #graph: [x, y, z,y_range, ut_y_error, scifi_hit_dist, abs(scifi_diff), detector_id]
    for ut_x, ut_y_begin, ut_y_end, ut_z in zip(track["ut_x"], track["ut_y_begin"], track["ut_y_end"], track["ut_z_position"]):
        ut_y_pred=y0+track["scifi_slope_dydz"]*(ut_z-z0)
        ut_y_range=abs(ut_y_end-ut_y_begin)

        #compute if ut_y_pred is in the ut_y_range
        if ut_y_pred<ut_y_begin:
            ut_y_error=ut_y_begin-ut_y_pred
        elif ut_y_pred>ut_y_end:
            ut_y_error=ut_y_pred-ut_y_end
        else:
            ut_y_error=0.0
        ut_y_error=ut_y_error/(ut_y_range+1e-8) #avoiding division by 0 if ut_y_range =0 (should be unless there is error in input data)

        node_ft.append([ut_x, ut_y_pred, ut_z, ut_y_error ,ut_y_range,0,det_type["UT"]])
        node_z.append(ut_z) #z - order of detector layers

    #least square method line in scifi segment in xz projection (no use in yz bc in yz points create straight line)
    scifi_diffs=[]
    scifi_x=track["scifi_x"]
    scifi_z=track["scifi_z"]

    z_mean=sum(scifi_z)/len(scifi_z)
    x_mean=sum(scifi_x)/len(scifi_x)

    num=0.0
    den=0.0
    for x, z in zip(scifi_x, scifi_z):
        num+=(z-z_mean)*(x- x_mean)
        den+=(z-z_mean)**2

    if den>0:
        a=num/den
        b=x_mean-a*z_mean
    else:
        a, b =0.0, x_mean


    for scifi_x, scifi_z in zip(track["scifi_x"], track["scifi_z"]):
        scifi_y=y0+track["scifi_slope_dydz"]*(scifi_z-z0)

        x_pred=a*scifi_z + b
        scifi_diff_1 = scifi_x-x_pred
        #scifi_diff_1=abs(scifi_diff_1)
        scifi_diffs.append(scifi_diff_1)

        node_ft.append([scifi_x,scifi_y, scifi_z,0,0,abs(scifi_diff_1),det_type["SciFi"]])
        node_z.append(scifi_z)

    scifi_diffs_all = (sum(r*r for r in scifi_diffs) / len(scifi_diffs)) ** 0.5
    track_tensor=torch.tensor(node_ft, dtype=torch.float)

    #z position in the next layers
    layers={}
    for i, z in enumerate(node_z):
        if z not in layers:
            layers[z]=[]
        layers[z].append(i)
    # for k, v in layers.items():
    #     logging.info("node %s: z=%s",v,k )

    #EDGES
    layer_z = list(layers.keys())

    edge_id=[]
    #edges connect nodes: z_before, z_after (two next z positions in layers)
    for z_bf, z_aft in zip(layer_z[:-1], layer_z[1:]):
        for i in layers[z_bf]:
            for j in layers[z_aft]:
                edge_id.append([i, j])
    edge_id=torch.tensor(edge_id, dtype=torch.long).t()
    #one element tensor (label true 1/ flase 0)
    label=torch.tensor([int(track["isDownstreamTrack"])], dtype=torch.long)

    #return Data(x=track_tensor, edge_index=edge_id, y=label) 
    # n_scifi_hits = len(track["scifi_z"])
    # data = Data(x=track_tensor, edge_index=edge_id, y=label)
    # data.graph_ft = torch.tensor([[scifi_diffs_all, n_scifi_hits]], dtype=torch.float) #not used - worse output
    # return data
    return Data(x=track_tensor, edge_index=edge_id, y=label)


#plot graph
def plot_graph_xz(graph, name, track_id):
    x=graph.x
    edge_index=graph.edge_index
    z=x[:, 2].numpy()
    x_x=x[:, 0].numpy()
    #nodes
    plt.scatter(z, x_x)
    for src, tgt in edge_index.t().numpy():
        plt.plot([z[src], z[tgt]], [x_x[src], x_x[tgt]])
    plt.xlabel("z [mm]")
    plt.ylabel("x [mm]")
    plt.title(f"Graf {track_id} w projekcji zx")
    plt.grid(True, alpha=0.3)
    os.makedirs("tracks_graphs_xz", exist_ok=True)
    filename=os.path.join("tracks_graphs_xz", name)
    plt.savefig(filename, dpi=150)
    plt.close()
    logging.info(f"Saved {filename}")

    #plot graph
def plot_graph_yz(graph, name, track_id):
    x=graph.x
    edge_index=graph.edge_index
    z=x[:, 2].numpy()
    y_y=x[:, 1].numpy()
    #nodes
    plt.scatter(z, y_y)
    for src, tgt in edge_index.t().numpy():
        plt.plot([z[src], z[tgt]], [y_y[src], y_y[tgt]])
    plt.xlabel("z [mm]")
    plt.ylabel("y [mm]")
    plt.title(f"Graf {track_id} w projekcji zy")
    plt.grid(True, alpha=0.3)
    os.makedirs("tracks_graphs_yz", exist_ok=True)
    filename=os.path.join("tracks_graphs_yz", name)
    plt.savefig(filename, dpi=150)
    plt.close()
    logging.info(f"Saved {filename}")

if __name__=='__main__':
    events =load_downstream_tracks("sample_small_new_data.csv")
    events =calc_avg_ut_xy_position(events)
    graph=build_graph_from_track(events[1])
    logging.info(graph)

    for i in range(12):
        track=events[i]
        graph=build_graph_from_track(events[i])
        plot_graph_xz(graph,f"track_{i:02d}.png",track_id=track["track_id"])
        plot_graph_yz(graph,f"track_{i:02d}.png", track_id=track["track_id"] )
    '''output: Data(x=[16, 6], edge_index=[2, 15], y=[1])
    Data (nodes) : 16 nodes, 6 node festures: #graph: [x, y, z,y_range, ut_y_error, detector_id]
    Edges : 2: source_node, target_node, 15 edges
    Label: 1: true/false
    '''


