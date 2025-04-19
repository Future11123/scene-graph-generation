import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from helper import getCoef

# 这个函数要大改
# def Gaussian2DLikelihood(outputs, targets, nodesPresent, pred_length):
#     """
#     Computes the likelihood of predicted locations under a bivariate Gaussian distribution
#     params:
#     outputs: Torch variable containing tensor of shape seq_length x numNodes x output_size #[20, 10, 5]
#     targets: Torch variable containing tensor of shape seq_length x numNodes x input_size #[20, 10, 2]最后一个数改为了3
#     nodesPresent : A list of lists, of size seq_length. Each list contains the nodeIDs that are present in the frame #len = 20
#     """
#     # 这里获得每一帧的节点id
#     nodesPresent = [[m[0] for m in t] for t in nodesPresent]
#     # Get the sequence length
#     seq_length = outputs.size()[0]
#     print(outputs)
#     print(targets)
#     # Get the observed length
#     obs_length = seq_length - pred_length
#
#     criterion = nn.CrossEntropyLoss()
#
#     # Compute the loss across all frames and all nodes
#     loss = 0
#     accuracy = 0
#     counter = 0
#     plot_datas = [[] for i in range(outputs.shape[1])]
#
#     for framenum in range(obs_length, seq_length):
#         nodeIDs = nodesPresent[framenum]
#
#         for nodeID in nodeIDs:
#
#             pred = outputs[framenum, nodeID, :]
#             target = targets[framenum, nodeID]
#
#             pred = pred.view(1,-1)
#             pred_y = torch.max(pred, 1)[1]
#             target = target.view(-1).long()
#             plot_data = [pred_y.data.cpu().numpy()[0], target.data.cpu().numpy()[0]]
#
#             plot_datas[nodeID].append(plot_data)
#             loss = loss + criterion(pred, target)
#             accuracy = accuracy + (pred_y == target).data.cpu().numpy()[0]
#
#             counter = counter + 1
#
#     if counter != 0:
#         return loss / counter, accuracy / counter, plot_datas
#     else:
#         return loss, accuracy, plot_datas
def Gaussian2DLikelihood(outputs, targets, nodesPresent, pred_length):
    """
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution
    params:
    outputs: Torch variable containing tensor of shape seq_length x numNodes x output_size #[20, 10, 5]
    targets: Torch variable containing tensor of shape seq_length x numNodes x input_size #[20, 10, 2]
    nodesPresent : A list of lists, of size seq_length. Each list contains the nodeIDs that are present in the frame #len = 20
    """
    # print(f"outputs shape: {outputs.shape}")
    # print(f"targets shape: {targets.shape}")
    # print(f"nodesPresent length: {len(nodesPresent)}")
    # print(f"pred_length: {pred_length}")

    nodesPresent = [[m[0] for m in t] for t in nodesPresent]
    # Get the sequence length
    seq_length = outputs.size()[0]
    # Get the observed length
    obs_length = seq_length - pred_length

    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy
    z = (
        torch.pow((normx / sx), 2)
        + torch.pow((normy / sy), 2)
        - 2 * ((corr * normx * normy) / sxsy)
    )
    negRho = 1 - torch.pow(corr, 2)

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))

    # Compute the loss across all frames and all nodes
    loss = 0
    counter = 0

    for framenum in range(obs_length, seq_length):
        nodeIDs = nodesPresent[framenum]

        for nodeID in nodeIDs:
            loss = loss + result[framenum, nodeID]
            counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss


"""
def Gaussian2DLikelihoodInference(outputs, targets, assumedNodesPresent, nodesPresent, use_cuda):
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution at test time
    params:
    outputs : predicted locations
    targets : true locations
    assumedNodesPresent : Nodes assumed to be present in each frame in the sequence
    nodesPresent : True nodes present in each frame in the sequence
    
    # Extract mean, std devs and correlation
    #assumedNodesPresent =
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))

    # Compute the loss
    loss = Variable(torch.zeros(1))
    if use_cuda:
        loss = loss.cuda()
    counter = 0

    for framenum in range(outputs.size()[0]):
        nodeIDs = nodesPresent[framenum]

        for nodeID in nodeIDs:
            if nodeID not in assumedNodesPresent:
                # If the node wasn't assumed to be present, don't compute loss for it
                continue
            loss = loss + result[framenum, nodeID]
            counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss       
"""
