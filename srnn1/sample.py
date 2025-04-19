import argparse
import os
import pickle
import time
import warnings
from picture import plot

import numpy as np
import torch
from IPython import embed
from torch.autograd import Variable

from criterion import Gaussian2DLikelihood
from helper import (
    compute_edges,
    get_final_error_separately,
    get_mean_error_separately,
    getCoef,
    sample_gaussian_2d,
)
from model import SRNN
from st_graph import ST_GRAPH
from utils import DataLoader

warnings.filterwarnings("ignore")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument(
        "--obs_length", type=int, default=20, help="Observed length of the trajectory"
    )
    # Predicted length of the trajectory parameter
    parser.add_argument(
        "--pred_length", type=int, default=1, help="Predicted length of the trajectory"
    )
    # Model to be loaded
    parser.add_argument(
        "--epoch", type=int, default=9, help="Epoch of model to be loaded"
    )

    # Use GPU or not
    parser.add_argument(
        "--use_cuda", action="store_true", default=True, help="Use GPU or CPU"
    )

    # Parse the parameters
    sample_args = parser.parse_args()

    # Save directory
    save_directory = "../save/"
    # Define the path for the config file for saved args
    with open(os.path.join(save_directory, "config.pkl"), "rb") as f:
        saved_args = pickle.load(f)

    # Initialize net
    net = SRNN(saved_args, True)
    if saved_args.use_cuda:
        net = net.cuda()

    checkpoint_path = os.path.join(
        save_directory, "srnn_model_" + str(sample_args.epoch) + ".tar"
    )

    if os.path.isfile(checkpoint_path):
        print("Loading checkpoint")
        checkpoint = torch.load(checkpoint_path)
        # model_iteration = checkpoint['iteration']
        model_epoch = checkpoint["epoch"]
        net.load_state_dict(checkpoint["state_dict"])
        print("Loaded checkpoint at {}".format(model_epoch))

    dataloader = DataLoader(
        1, sample_args.pred_length + sample_args.obs_length, True, infer=True
    )

    dataloader.reset_batch_pointer()

    # Construct the ST-graph object
    stgraph = ST_GRAPH(1, sample_args.pred_length + sample_args.obs_length)

    results = []

    # Variable to maintain total error
    # total_error = 0
    # final_error = 0
    avg_ped_acc = 0
#    final_ped_error = 0
    avg_bic_acc = 0
#    final_bic_error = 0
    avg_car_acc = 0
#    final_car_error = 0
    ped_total_counter = 0
    bic_total_counter = 0
    car_total_counter = 0

    for batch in range(dataloader.num_batches):
        start = time.time()

        # Get the next batch
        x, _, frameIDs, d = dataloader.next_batch(randomUpdate=False)

        # Construct ST graph
        stgraph.readGraph(x)

        nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()

        # Convert to cuda variables
        nodes = Variable(torch.from_numpy(nodes).float(), volatile=True)
        edges = Variable(torch.from_numpy(edges).float(), volatile=True)
        if saved_args.use_cuda:
            nodes = nodes.cuda()
            edges = edges.cuda()

        # Separate out the observed part of the trajectory
        obs_nodes, obs_edges, obs_nodesPresent, obs_edgesPresent = (
            nodes[: sample_args.obs_length,:,:],
            edges[: sample_args.obs_length],
            nodesPresent[: sample_args.obs_length],
            edgesPresent[: sample_args.obs_length],
        )

        # Sample function
        ret_nodes, ret_attn, datas = sample(
            obs_nodes,
            obs_edges,
            obs_nodesPresent,
            obs_edgesPresent,
            sample_args,
            net,
            nodes,
            edges,
            nodesPresent,
        )
        
#        画图
#        save_dir = './train_plot'
#        save_name = str(batch)
#        plot(save_dir, save_name, datas)

        # Compute mean and final displacement error
        """
        total_error += get_mean_error(
            ret_nodes[sample_args.obs_length :].data,
            nodes[sample_args.obs_length :].data,
            nodesPresent[sample_args.obs_length - 1],
            nodesPresent[sample_args.obs_length :],
            saved_args.use_cuda,
        )
        final_error += get_final_error(
            ret_nodes[sample_args.obs_length :].data,
            nodes[sample_args.obs_length :].data,
            nodesPresent[sample_args.obs_length - 1],
            nodesPresent[sample_args.obs_length :],
        )
        """
        avg_ped_acc_delta, avg_bic_acc_delta, avg_car_acc_delta, ped_counter, bic_counter, car_counter = get_mean_error_separately(
            ret_nodes[sample_args.obs_length :].data,
            nodes[sample_args.obs_length :].data,
            nodesPresent[sample_args.obs_length - 1],
            nodesPresent[sample_args.obs_length :],
            saved_args.use_cuda,
        )
        ped_total_counter += ped_counter
        bic_total_counter += bic_counter
        car_total_counter += car_counter
        avg_ped_acc += avg_ped_acc_delta
        avg_bic_acc += avg_bic_acc_delta
        avg_car_acc += avg_car_acc_delta

#        final_ped_error_delta, final_bic_error_delta, final_car_error_delta = get_final_error_separately(
#            ret_nodes[sample_args.obs_length :].data,
#            nodes[sample_args.obs_length :].data,
#            nodesPresent[sample_args.obs_length - 1],
#            nodesPresent[sample_args.obs_length :],
#        )
#        final_ped_error += final_ped_error_delta
#        final_bic_error += final_bic_error_delta
#        final_car_error += final_car_error_delta

        end = time.time()

        print(
            "Processed trajectory number : ",
            batch,
            "out of",
            dataloader.num_batches,
            "trajectories in time",
            end - start,
        )
        if saved_args.use_cuda:
            results.append(
                (
                    nodes.data.cpu().numpy(),
                    ret_nodes.data.cpu().numpy(),
                    nodesPresent,
                    sample_args.obs_length,
                    ret_attn,
                    frameIDs,
                )
            )
        else:
            results.append(
                (
                    nodes.data.numpy(),
                    ret_nodes.data.numpy(),
                    nodesPresent,
                    sample_args.obs_length,
                    ret_attn,
                    frameIDs,
                )
            )

        # Reset the ST graph
        stgraph.reset()

    # print("Total mean error of the model is ", total_error / dataloader.num_batches)
    # print(
    #    "Total final error of the model is ", final_error / dataloader.num_batches
    # )  # num_batches = 10
    print(
        "AVG disp acc:     pedestrian: {}       bicycle: {}        car:{}".format(
            avg_ped_acc / ped_total_counter,
            avg_bic_acc / bic_total_counter,
            avg_car_acc / car_total_counter,
        )
    )

    print(
        "total average acc:    {}".format(
            (avg_ped_acc + avg_bic_acc + avg_car_acc) / (ped_total_counter + bic_total_counter + car_total_counter)
        )
    )

#    print(
#        "Final disp error:   pedestrian: {}       bicycle: {}        car:{}".format(
#            final_ped_error / dataloader.num_batches,
#            final_bic_error / dataloader.num_batches,
#            final_car_error / dataloader.num_batches,
#        )
#    )
#    print(
#        "total final error:    {}".format(
#            (final_ped_error + final_bic_error + final_car_error)
#            / dataloader.num_batches
#            / 3
#        )
#    )

    print("Saving results")
    with open(os.path.join(save_directory, "results.pkl"), "wb") as f:
        pickle.dump(results, f)


def sample(
    nodes,
    edges,
    nodesPresent,
    edgesPresent,
    args,
    net,
    true_nodes,
    true_edges,
    true_nodesPresent,
):
    """
    Sample function
    Parameters
    ==========

    nodes : A tensor of shape obs_length x numNodes x 2
    Each row contains (x, y)

    edges : A tensor of shape obs_length x numNodes x numNodes x 2
    Each row contains the vector representing the edge
    If edge doesn't exist, then the row contains zeros

    nodesPresent : A list of lists, of size obs_length
    Each list contains the nodeIDs that are present in the frame

    edgesPresent : A list of lists, of size obs_length
    Each list contains tuples of nodeIDs that have edges in the frame

    args : Sampling Arguments

    net : The network

    Returns
    =======

    ret_nodes : A tensor of shape (obs_length + pred_length) x numNodes x 2
    Contains the true and predicted positions of all the nodes
    """
    # Number of nodes
    numNodes = nodes.size()[1]

    # Initialize hidden states for the nodes
    h_nodes = Variable(torch.zeros(numNodes, net.args.node_rnn_size), volatile=True)
    h_edges = Variable(
        torch.zeros(numNodes * numNodes, net.args.edge_rnn_size), volatile=True
    )
    c_nodes = Variable(torch.zeros(numNodes, net.args.node_rnn_size), volatile=True)
    c_edges = Variable(
        torch.zeros(numNodes * numNodes, net.args.edge_rnn_size), volatile=True
    )
    h_super_node = Variable(torch.zeros(3, net.args.node_rnn_size), volatile=True)
    c_super_node = Variable(torch.zeros(3, net.args.node_rnn_size), volatile=True)
    h_super_edges = Variable(torch.zeros(3, net.args.edge_rnn_size), volatile=True)
    c_super_edges = Variable(torch.zeros(3, net.args.edge_rnn_size), volatile=True)
    if args.use_cuda:
        h_nodes = h_nodes.cuda()
        h_edges = h_edges.cuda()
        c_nodes = c_nodes.cuda()
        c_edges = c_edges.cuda()
        h_super_node = h_super_node.cuda()
        c_super_node = c_super_node.cuda()
        h_super_edges = h_super_edges.cuda()
        c_super_edges = c_super_edges.cuda()

    # Propagate the observed length of the trajectory
    for tstep in range(args.obs_length - 1):
        # Forward prop
        out_obs, h_nodes, h_edges, c_nodes, c_edges, h_super_node, h_super_edges, c_super_node, c_super_edges, _ = net(
            nodes[tstep].view(1, numNodes, 3),
            edges[tstep].view(1, numNodes * numNodes, 2),
            [nodesPresent[tstep]],
            [edgesPresent[tstep]],
            h_nodes,
            h_edges,
            c_nodes,
            c_edges,
            h_super_node,
            h_super_edges,
            c_super_node,
            c_super_edges,
        )
        # loss_obs = Gaussian2DLikelihood(out_obs, nodes[tstep+1].view(1, numNodes, 2), [nodesPresent[tstep+1]])

    # Initialize the return data structures
    ret_nodes = Variable(
        torch.zeros(args.obs_length + args.pred_length, numNodes, 3), volatile=True
    )
    if args.use_cuda:
        ret_nodes = ret_nodes.cuda()
    ret_nodes[: args.obs_length, :, :] = nodes.clone()

    ret_edges = Variable(
        torch.zeros((args.obs_length + args.pred_length), numNodes * numNodes, 2),
        volatile=True,
    )
    if args.use_cuda:
        ret_edges = ret_edges.cuda()
    ret_edges[: args.obs_length, :, :] = edges.clone()

    ret_attn = []

    # Propagate the predicted length of trajectory (sampling from previous prediction)
    for tstep in range(args.obs_length - 1, args.pred_length + args.obs_length - 1):
        # TODO Not keeping track of nodes leaving the frame (or new nodes entering the frame, which I don't think we can do anyway)
        # Forward prop
        outputs, h_nodes, h_edges, c_nodes, c_edges, h_super_node, h_super_edges, c_super_node, c_super_edges, attn_w = net(
            ret_nodes[tstep].view(1, numNodes, 3),
            ret_edges[tstep].view(1, numNodes * numNodes, 2),
            [nodesPresent[args.obs_length - 1]],
            [edgesPresent[args.obs_length - 1]],
            h_nodes,
            h_edges,
            c_nodes,
            c_edges,
            h_super_node,
            h_super_edges,
            c_super_node,
            c_super_edges,
        )
#        mux, muy, sx, sy, corr = getCoef(outputs)
        next_inter = sample_gaussian_2d(
            outputs,
            nodesPresent[args.obs_length - 1],
        )

        ret_nodes[tstep + 1, :, 2] = next_inter
#        ret_nodes[tstep + 1, :, 0] = next_x
#        ret_nodes[tstep + 1, :, 1] = next_y

        # Compute edges
        # TODO Currently, assuming edges from the last observed time-step will stay for the entire prediction length
        ret_edges[tstep + 1, :, :] = compute_edges(
            ret_nodes.data, tstep + 1, edgesPresent[args.obs_length - 1], args.use_cuda
        )
        # Store computed attention weights
        ret_attn.append(attn_w[0])
        
    #画图数据
    nodesPresent = [[m[0] for m in t] for t in true_nodesPresent]
    nodesPresent_type = [[m[1] for m in t] for t in true_nodesPresent]
    plot_datas = [[] for i in range(true_nodes.shape[1])]
    for frame in range(args.obs_length, args.obs_length + args.pred_length):
        nodeIDs = nodesPresent[frame]
        types = nodesPresent_type[frame]
        
        for k,nodeID in enumerate(nodeIDs):
            pred = ret_nodes[frame, nodeID, 2]
            target = true_nodes[frame, nodeID, 2]
            
            plot_data = [pred.data.item(), target.data.item(), types[k]]
            plot_datas[nodeID].append(plot_data)

    return ret_nodes, ret_attn, plot_datas


if __name__ == "__main__":
    main()
