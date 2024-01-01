import torch
import torch.nn as nn
import numpy as np
import torchvision

    
class Dist_wrapper(nn.Module):
    def __init__(self, grid_size=64, gamma = 200):
        super(Dist_wrapper, self).__init__()
        self.gamma = gamma
        self.grid_sqz = self.make_coordinate_grid(grid_size)
        self.dist_transform = self.distance_transform

    

    def make_coordinate_grid(self,dim, scale=1):
        """
        Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
        """
        h, w = (dim, dim)
        x = torch.arange(w).float()
        y = torch.arange(h).float()

        yy = y.view(-1, 1).repeat(1, w)
        xx = x.view(1, -1).repeat(h, 1)

        meshed = (torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2).unsqueeze(0).unsqueeze(0) / (h - 1) * 2) - 1
        meshed *= scale

        return meshed

    def distance_transform(self, kps, grid_sqz, gamma=3., glb=False, mask=None):
        grid_size = grid_sqz.shape[-2]
        grid_sqz = grid_sqz.repeat(kps.size(0),1, 1, 1, 1)
        # kps = torch.index_select(kps_, 2, torch.LongTensor([1, 0]).to(kps_.device))
        pi_set = kps[:, :-1].unsqueeze(-2).unsqueeze(-2)
        pj_set = kps[:, 1:].unsqueeze(-2).unsqueeze(-2)

        # Compute r
        v_set = (pi_set - pj_set).repeat(1, 1, grid_size, grid_size, 1)
        v_norm = (v_set.pow(2)).sum(-1).unsqueeze(-1)
        u_set = (grid_sqz - pj_set)

        uv = torch.bmm(u_set.view(-1, 1, 2), (v_set).view(-1, 2, 1)).view(kps.shape[0], -1, grid_size, grid_size, 1)
        rs = torch.clamp(uv / v_norm, 0, 1)#.detach()
        rs.masked_scatter_(rs.isnan(), uv)
        
        betas = ((u_set - rs * v_set).pow(2)).sum(-1)
        betas = torch.exp(-gamma * betas)

        
        if mask is not None:
            betas = betas * (~mask[:, :-1]).float().unsqueeze(-1).unsqueeze(-1)
        


        betas = betas.max(1)[0]
        if glb:
            betas = betas.max(0)[0]

        return betas


    def forward_single(self, pred_coords, gt_coords, gamma = None, gw = False):
        
        # Pred_Coords = (N1, 3)
        # GT_Coords = (N2, 3)
        # Mask_Pred = (N1, )
        # Mask_GT = (N2, )
        if gamma is None:
            gamma = self.gamma
        
        
        grid_sqz_pred = self.grid_sqz.repeat(1,pred_coords.shape[0]-1,  1, 1, 1).cuda()
        grid_sqz_gt = self.grid_sqz.repeat(1,gt_coords.shape[0]-1,  1, 1, 1).cuda()
        
        mask_gt = gt_coords[:, -1].type(torch.bool).cuda()
        gt_coords_curr = (gt_coords[:,:-1] - 0.5)*2.0
        
        mask_pred = (pred_coords[:,-1]>0.5).type(torch.bool).cuda()
        pred_coords_curr = (pred_coords[:,:-1] - 0.5)*2.0
        

        gt_dist = self.dist_transform(gt_coords_curr.unsqueeze(0), grid_sqz_gt,mask = mask_gt.unsqueeze(0), gamma = gamma)
        
        pred_dist = self.dist_transform(pred_coords_curr.unsqueeze(0), grid_sqz_pred,mask = mask_pred.unsqueeze(0), gamma = gamma)
        dist_loss = torch.nn.functional.mse_loss(pred_dist.float(), gt_dist.float()).float().mean()

                
        return {"dist_loss": dist_loss}
        
    