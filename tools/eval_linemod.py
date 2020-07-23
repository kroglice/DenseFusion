import argparse
import os
import numpy as np
import yaml
import torch
import torch.nn.parallel
import torch.utils.data

from torch.autograd import Variable
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
# from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from transformations import quaternion_matrix
from lib.knn.__init__ import KNearestNeighbor

from lib.utils import iterative_points_refine

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
opt = parser.parse_args()


def main():
    num_objects = 13
    objlist = [1]
    num_points = 500
    iteration = 4
    bs = 1
    dataset_config_dir = os.path.join(opt.dataset_root, 'models')
    output_result_dir = os.path.join(os.getcwd(), 'experiments/eval_result/linemod')
    if not os.path.exists(output_result_dir):
        os.makedirs(os.path.join(output_result_dir))
    knn = KNearestNeighbor(1)

    estimator = PoseNet(num_points = num_points, num_obj = num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
    refiner.cuda()
    estimator.load_state_dict(torch.load(opt.model))
    refiner.load_state_dict(torch.load(opt.refine_model))
    estimator.eval()
    refiner.eval()

    testdataset = PoseDataset_linemod('eval', num_points, False, opt.dataset_root, 0.0, True, objlist)
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=bs, shuffle=False, num_workers=10)

    sym_list = testdataset.get_sym_list()
    num_points_mesh = testdataset.get_num_points_mesh()
    criterion = Loss(num_points_mesh, sym_list)
    criterion_refine = Loss_refine(num_points_mesh, sym_list)

    diameter = []
    meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
    meta = yaml.load(meta_file)
    for obj in objlist:
        diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
    print(diameter)

    success_count = [0 for i in range(num_objects)]
    num_count = [0 for i in range(num_objects)]
    fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

    for i, data in enumerate(testdataloader, 0):
        points, choose, img, target, model_points, idx = data
        # cropped pointcloud, chosen_pixel_mask, normalized_cropped_img, target (transformed model_points),
        # model_points (sampled points from object mesh), object index
        if len(points.size()) == 2:
            print('No.{0} NOT Pass! Lost detection!'.format(i))
            fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
            continue
        points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                         Variable(choose).cuda(), \
                                                         Variable(img).cuda(), \
                                                         Variable(target).cuda(), \
                                                         Variable(model_points).cuda(), \
                                                         Variable(idx).cuda()

        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)

        # get the rotation and translation of the most confident predicted point from the cloud set
        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)

        my_pred, my_r, my_t = iterative_points_refine (refiner, points, emb, idx, iteration, my_r, my_t, bs, num_points)
        # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

        model_points = model_points[0].cpu().detach().numpy()
        my_r = quaternion_matrix(my_r)[:3, :3]
        pred = np.dot(model_points, my_r.T) + my_t  # prediction of sampled pixel after being transformed
        target = target[0].cpu().detach().numpy()

        if idx[0].item() in sym_list:
            pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
            target = torch.index_select(target, 1, inds.view(-1) - 1)
            dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
        else:
            dis = np.mean(np.linalg.norm(pred - target, axis=1))    # Eq. (1)

        if dis < diameter[idx[0].item()]:
            success_count[idx[0].item()] += 1
            print('No.{0} Pass! Distance: {1}'.format(i, dis))
            fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
        else:
            print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
            fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
        num_count[idx[0].item()] += 1

    for i in range(num_objects):
        print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
        fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
    print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
    fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
    fw.close()


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()