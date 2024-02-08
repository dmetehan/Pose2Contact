import logging
import os.path
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def vis_threshold_eval(gts, scores, eval_func, epoch, save_dir, **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    xs = torch.linspace(0, 1, 25)
    for key in gts:
        eval_results = [eval_func(gts[key], torch.Tensor(scores[key]) > thresh, **kwargs) for thresh in xs]
        plt.plot(xs, eval_results, label=key)
    plt.xticks(torch.linspace(0, 1, 25))
    plt.xticks(rotation=90)
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(save_dir, f'{epoch}.png'))
    plt.clf()


def _init_heatmaps():
    people = ['adult', 'child']
    sketches = {pers: cv2.imread('data/rid_base.png') for pers in people}

    def create_mask_index():
        mask_ind = {pers: np.zeros_like(cv2.imread(f'data/rid_base.png', 0)) for pers in people}
        for pers in people:
            for rid in range(21):
                mask = cv2.imread(f'data/masks_coarse_clear/rid_{rid}.png', 0)
                mask_ind[pers][mask < 100] = rid + 1
        return mask_ind

    return people, sketches, create_mask_index()


def vis_pred_errors_heatmap(gts, preds, save_dir):
    logging.info(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    gts = torch.Tensor(gts) == 1
    xor = torch.bitwise_xor(gts, preds)
    false_positives = torch.bitwise_and(preds, xor).sum(dim=0)
    false_negatives = torch.bitwise_and(gts, xor).sum(dim=0)
    false_all = false_positives + false_negatives
    minimum, maximum = min(false_all), max(false_all)
    people, sketches, mask_ind = _init_heatmaps()
    false_all = {'adult': false_all[:21], 'child': false_all[21:]}
    for person in people:
        img_cpy = sketches[person].copy()
        colormap = plt.get_cmap('RdYlGn')
        for regions, count in enumerate(false_all[person]):
            x = (count - minimum) / (maximum - minimum)
            r, g, b = colormap(float(1-x))[:3]  # reverse colormap red to green
            img_cpy[mask_ind[person] == (regions + 1)] = (255 * b, 255 * g, 255 * r)
        cv2.imwrite(os.path.join(save_dir, f'{person}.png'), img_cpy)
        cv2.imshow(person, img_cpy)
    cv2.waitKey(0)
