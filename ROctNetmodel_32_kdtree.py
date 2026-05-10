import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time
import random

class GaussianSplat(nn.Module):
    #
    #      xyz (Tensor): shape (N,3) 3D positions
    #    scale (Tensor): shape (N,3) gaussian scales
    #    rotation (Tensor): shape (N,4) quaternion rotations
    #    opacity (Tensor): shape (N,1) opacity values
    #    features (Tensor): shape (N,C) color or SH features
    #
    def __init__(self, xyz, scale, rotation, opacity, feature):
        super(GaussianSplat, self).__init__()
        self.xyz = xyz
        self.scale = scale
        self.rotation = rotation
        self.opacity = opacity
        self.feature = feature
    
    def forward(self, *input):
        
        return {
            "xyz": self.xyz,
            "scale":self.scale,
            "rotation": self.rotation,
            "opacity": self.opacity,
            "feature": self.feature
        }
        

#########################################################################################
## Encoder
#########################################################################################
class Sampler(nn.Module):

    def __init__(self, hidden_size):
        super(Sampler, self).__init__()
        self.Linear1 = nn.Linear(hidden_size, hidden_size *2)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, r_node):
        #print(input.size())
        output = self.Linear1(r_node)
        output = self.relu(output)
        output = self.Linear2(output)
        return output

class LeafEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LeafEncoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear_count = nn.Linear(1, hidden_size)
        self.linear2 = nn.Linear(hidden_size * 3, hidden_size)
        self.relu2 = nn.ReLU()

    def forward(self, leaf):
        device = self.linear1.weight.device
        points = getattr(leaf, "points_tensor", None)
        if points is None and torch.is_tensor(leaf.points):
            points = leaf.points
        if points is not None:
            if points.dim() == 1:
                points = points.unsqueeze(0)
            points = points.to(device)
            feats = self.linear1(points)
            feats = self.relu1(feats)
        else:
            # Fallback: stack once to avoid per-point tensor creation
            pts = leaf.points
            if not torch.is_tensor(pts):
                pts = torch.as_tensor(pts, dtype=torch.float32, device=device)
            if pts.dim() == 1:
                pts = pts.unsqueeze(0)
            leaf.points_tensor = pts
            feats = self.linear1(pts)
            feats = self.relu1(feats)
        max_pool = torch.max(feats, dim=0)[0]
        mean_pool = torch.mean(feats, dim=0)
        count = torch.tensor([[float(len(leaf.points))]], dtype=feats.dtype, device=feats.device)
        count_feat = self.linear_count(count).squeeze(0)
        merged = torch.cat([max_pool, mean_pool, count_feat], dim=0).unsqueeze(0)
        merged = self.linear2(merged)
        merged = self.relu2(merged)
        return merged
        

class NodeEncoder(nn.Module):

    def __init__(self, hidden_size):
        super(NodeEncoder, self).__init__()
        self.linear1 = nn.Linear(hidden_size * 2 + 3, hidden_size * 2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size)

        
    def forward(self, leaf_left, leaf_right, axis):
        axis_index = int(axis.item()) if torch.is_tensor(axis) else int(axis)
        axis_vec = torch.zeros(1, 3, device=leaf_left.device)
        axis_vec[0, axis_index] = 1.0
        merged = torch.cat([leaf_left, leaf_right, axis_vec], dim=1)
        output = self.linear1(merged)
        output = self.relu1(output)
        return self.linear2(output)



class ROctEncoder(nn.Module):

    def __init__(self, config):
        super(ROctEncoder, self).__init__()
        self.leaf_encoder = LeafEncoder(input_size = config.leaf_code_size, hidden_size = config.hidden_size)
        
        self.node_encoder = NodeEncoder(hidden_size = config.hidden_size)
 
        self.sample_encoder = Sampler(hidden_size = config.hidden_size)


    def LeafEncoder(self, node):
        return self.leaf_encoder(node)
    
    def NodeEncoder(self, node1, node2, axis):
        return self.node_encoder(node1, node2, axis)
    
    def sampleEncoder(self, node):
        return self.sample_encoder(node)
   
def encode_structure(tree, encoder):
    def encode_node(node):
        if node is None:
            return None
        if node.is_leaf():
            return encoder.LeafEncoder(node)
        child1 = encode_node(node.left)
        child2 = encode_node(node.right)
        axis = torch.tensor([node.axis], dtype=torch.long, device=child1.device)
        return encoder.NodeEncoder(child1, child2, axis)

    encoding = encode_node(tree.root)
    return encoder.sampleEncoder(encoding)

#########################################################################################
## Decoder
#########################################################################################

class SampleDecoder(nn.Module):
    """ Decode a randomly sampled noise into a feature vector """
    def __init__(self, hidden_size):
        super(SampleDecoder, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, input_feature):
        output = self.linear1(input_feature)
        output = self.relu(output)
        return self.linear2(output)


class NodeDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(NodeDecoder, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size * 2,  hidden_size * 2)
        

    def forward(self, parent_feature):
        vector = self.linear1(parent_feature)
        vector = self.relu(vector)
        vector = self.linear2(vector)
        return vector.chunk(2, dim=1)

class leafDecoder(nn.Module):
    
    def __init__(self, feature_size, hidden_size, point_count=4):
        super(leafDecoder, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.point_count = point_count
        self.linear2 = nn.Linear(hidden_size, feature_size * point_count)
        self.scores_head = nn.Linear(hidden_size, point_count)

    def forward(self, leaf_input):
        input = self.linear1(leaf_input)
        input = self.relu(input)
        feat = self.linear2(input)
        score = self.scores_head(input)
        return feat, score

class ROctDecoder(nn.Module):
    def __init__(self, config):
        super(ROctDecoder, self).__init__()
        self.leaf_decoder = leafDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size, point_count = config.leaf_point_count)
        self.node_decoder = NodeDecoder( hidden_size = config.hidden_size)
        self.sample_decoder = SampleDecoder( hidden_size = config.hidden_size)
        self.mseLoss = nn.MSELoss()  
        self.creLoss = nn.CrossEntropyLoss()  
        self.loss_weights = getattr(config, "loss_weights", None)
        self._loss_weights_tensor = None

    def leafDecoder(self, node):
        return self.leaf_decoder(node)

    def NodeDecoder(self, node):
        return self.node_decoder(node)

    def sampleDecoder(self, node):
        return self.sample_decoder(node)

 
    def leafLossEstimator(self, leaf_feature, gt_leaf_feature, pred_mask=None):
        if gt_leaf_feature.dim() == 1:
            gt_leaf_feature = gt_leaf_feature.unsqueeze(0)
        if leaf_feature.dim() == 1:
            leaf_feature = leaf_feature.unsqueeze(0)
        feat_size = gt_leaf_feature.size(1)
        pred = leaf_feature.view(-1, self.leaf_decoder.point_count, feat_size)
        gt = gt_leaf_feature
        mask = pred_mask
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            mask = mask.to(dtype=pred.dtype, device=pred.device).clamp_min(1e-6)

        if self.loss_weights is not None:
            if self._loss_weights_tensor is None or self._loss_weights_tensor.numel() != feat_size:
                w = torch.tensor(self.loss_weights, dtype=pred.dtype, device=pred.device)
                if w.numel() != feat_size:
                    raise ValueError(f"loss_weights length {w.numel()} != feature size {feat_size}")
                self._loss_weights_tensor = w
            w = self._loss_weights_tensor
            pred = pred * w
            gt = gt * w

    
        pred_exp = pred.unsqueeze(2)  # (B, K, 1, C)
        gt_exp = gt.unsqueeze(0).unsqueeze(1)  # (1, 1, N, C)
        dists = torch.sum((pred_exp - gt_exp) ** 2, dim=-1)  # (B, K, N)
        pred_to_gt = torch.min(dists, dim=2)[0]
        if mask is None:
            loss1 = pred_to_gt.mean()
            loss2 = torch.min(dists, dim=1)[0].mean()
        else:
            loss1 = (pred_to_gt * mask).sum() / mask.sum().clamp_min(1.0)
            active_dists = dists / mask.unsqueeze(2)
            loss2 = torch.min(active_dists, dim=1)[0].mean()
        loss = loss1 + loss2
        return loss.unsqueeze(0)

    def classifyLossEstimator(self, label_vector, gt_label_vector):

        loss = torch.cat([self.creLoss(l.unsqueeze(0), gt).mul(10) for l, gt in zip(label_vector, gt_label_vector)], 0)

        return loss

    def vectorAdder(self, v1, v2):
            return v1.add_(v2)
        
    def vectorAdder2(self, v1,v2):
        return v1.add_(v2)
    
    def vectorZeros(self, v):
        return v.mul_(0)


def decode_structure(feature, tree, decoder, keep_list=None, temperature=0.5, reduction="sum"):
    def decode_node_leaf(node, feat):
        if node.is_leaf():
            fea, score = decoder.leafDecoder(feat)
            valid_keep = keep_list
            if valid_keep is None:
                valid_keep = [
                    max(1, decoder.leaf_decoder.point_count // 4),
                    max(1, decoder.leaf_decoder.point_count // 2),
                    decoder.leaf_decoder.point_count,
                ]
            valid_keep = [k for k in valid_keep if 1 <= k <= decoder.leaf_decoder.point_count]
            if not valid_keep:
                valid_keep = [decoder.leaf_decoder.point_count]
            k = random.choice(valid_keep)
            threshold = torch.topk(score, k=k, dim=1).values[:, -1:]  # (B, 1)
            mask = torch.sigmoid((score - threshold) / temperature)
            gt = getattr(node, "points_tensor", None)
            if gt is None:
                gt = torch.as_tensor(node.points, dtype=torch.float32, device=fea.device)
                if gt.dim() == 1:
                    gt = gt.unsqueeze(0)
                node.points_tensor = gt
            else:
                gt = gt.to(fea.device)
            recon_loss_all = decoder.leafLossEstimator(fea, gt)
            recon_loss_k = decoder.leafLossEstimator(fea, gt, pred_mask=mask)
            leaf_count = torch.ones_like(recon_loss_all)
            return (recon_loss_all + recon_loss_k) / 2.0, leaf_count
        child1, child2 = decoder.NodeDecoder(feat)

        child_loss1, child_count1 = decode_node_leaf(node.left, child1)
        child_loss2, child_count2 = decode_node_leaf(node.right, child2)
        loss = decoder.vectorAdder(child_loss1, child_loss2)
        count = decoder.vectorAdder(child_count1, child_count2)
        return loss, count

    feature = decoder.sampleDecoder(feature)
    total_loss, total_leaf = decode_node_leaf(tree.root, feature)
    if reduction == "mean":
        return total_loss / total_leaf.clamp_min(1.0)
    if reduction == "sum":
        return total_loss
    raise ValueError(f"Unsupported reduction: {reduction}")
