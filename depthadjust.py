import numpy as np
import open3d as o3d
from math import floor, ceil
from scipy.interpolate import griddata

def split(length, ratio=0.8):    
    np.random.seed(42)
    indices = np.arange(length)
    np.random.shuffle(indices)
    train_indices = indices[:int(len(indices)*ratio)]
    test_indices = indices[int(len(indices)*ratio):]
    return train_indices, test_indices

def ransac_for_scale(z1, z2, iter_num=1000, thres=50, eps=1e-7):
    scale = None
    valid_points = 0
    for i in range(iter_num):
        idx1, idx2 = np.random.randint(0, len(z1), size=(2))
        if idx1 == idx2: continue
        tmp_scale = abs((z1[idx1] - z1[idx2] + eps) / (z2[idx1] - z2[idx2] + eps))
        tmp_valid_msk = (z2 * tmp_scale - z1 < thres)
        tmp_valid_points = tmp_valid_msk.sum()
        if tmp_valid_points > valid_points:
            scale = tmp_scale
            tmp_valid_points = valid_points
        if valid_points > (len(z1) // 2):
            break
    return scale
    
class DepthAdjust:
    def __init__(self):
        self.src_scalef = 1.
        self.tar_scalef = 1.
        self.transformation = np.eye(4)
        
    def get_pseudo_intrinsic(self, h, w):
        h = h - 1
        w = w - 1
        intrinsic = [
            [w, 0, w/2],
            [0, w, h/2],
            [0, 0, 1]
        ]
        return np.array(intrinsic)
    
    def fit_and_trainsform(self, src, tar, mode='q3', icp=False, return_proj=False):
        src_3d = self.depth23d(src)
        tar_3d = self.depth23d(tar)

        valid_indice = tar_3d[...,2] > 0.
        src_3d = src_3d[valid_indice]
        tar_3d = tar_3d[valid_indice]
        train_indice, test_indice = split(len(src_3d))
        self.solve_3dp2p(src_3d[train_indice], tar_3d[train_indice], mode, icp)
        
        if not icp:
            src_d = src_3d[test_indice][:,2]
            tar_d = tar_3d[test_indice][:,2]
            if return_proj:
                return src_d / self.src_scalef * self.tar_scalef, tar_d, (src / self.src_scalef * self.tar_scalef).squeeze()
            else:
                return src_d, src_d / self.src_scalef * self.tar_scalef, tar_d
        else:
            h, w = src.shape[:2]
            src_3d /= self.src_scalef
            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector(self.depth23d(src).reshape(-1, 3))
            src_pcd.transform(self.transformation)
            src_pcd.scale(self.tar_scalef, np.array([[0],[0],[0]]))
            np_points = np.asarray(src_pcd.points)
            z = np_points[...,2].copy()
            np_points /= np.expand_dims(z, -1)
            intrinsic = self.get_pseudo_intrinsic(h, w)
            recover_points = np.einsum('ni, di->nd', np_points, intrinsic)
            reproject = griddata(recover_points[...,:2], z, np.mgrid[0:w, 0:h].transpose(), method='linear', fill_value=0)
            # return reproject
            valid_z = reproject[valid_indice]
            test_z = valid_z[test_indice]
            if return_proj:
                return test_z, tar_3d[test_indice][:,2], reproject
            else:
                return test_z, tar_3d[test_indice][:,2]
            
    def solve_3dp2p(self, src_3d, tar_3d, mode='q3', icp=False):
        if mode == 'q3':
            self.src_scalef = np.quantile(src_3d[:,2], 0.75)
            self.tar_scalef = np.quantile(tar_3d[:,2], 0.75)
        elif mode == 'mean':
            self.src_scalef = np.mean(src_3d[:,2])
            self.tar_scalef = np.mean(tar_3d[:,2])
        elif mode == 'q1_q3':
            length = len(src_3d)
            self.src_scalef = np.sort(src_3d)[floor(length*0.25):ceil(length*0.75)].mean()
            self.tar_scalef = np.sort(tar_3d)[floor(length*0.25):ceil(length*0.75)].mean()
        elif mode == 'relative':
            self.src_scalef = ransac_for_scale(src_3d[:,2], tar_3d[:,2])
            self.tar_scalef = 1.0
            
        if icp:
            src_3d = src_3d.copy()
            tar_3d = tar_3d.copy()
            src_3d /= self.src_scalef
            tar_3d /= self.tar_scalef
            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector(src_3d)
            tar_pcd = o3d.geometry.PointCloud()
            tar_pcd.points = o3d.utility.Vector3dVector(tar_3d)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                src_pcd, tar_pcd, 0.005,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
            )
            self.transformation = reg_p2p.transformation
        
    def depth23d(self, depth):
        h, w = depth.shape[:2]
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        zz = np.ones_like(xx)
        axis = np.concatenate(
            [
                np.expand_dims(xx, -1),
                np.expand_dims(yy, -1),
                np.expand_dims(zz, -1)
            ],
            axis=-1
        ).astype(np.float64)
        point_3d = np.einsum('hwi, di->hwd', axis, np.linalg.inv(self.get_pseudo_intrinsic(h, w)))
        point_3d *= depth
        return point_3d
