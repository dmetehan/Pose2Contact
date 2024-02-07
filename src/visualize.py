import os.path

import torch
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

