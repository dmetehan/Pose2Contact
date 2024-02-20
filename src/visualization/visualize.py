import json
import logging
import os.path
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from sklearn import manifold, datasets
from sklearn.decomposition import TruncatedSVD
import plotly.express as px


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
    true_positives = torch.bitwise_and(gts, preds).sum(dim=0)
    xor = torch.bitwise_xor(gts, preds)
    false_positives = torch.bitwise_and(preds, xor).sum(dim=0)
    false_negatives = torch.bitwise_and(gts, xor).sum(dim=0)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)
    # minimum, maximum = ({'precision': min(precision), 'recall': min(recall), 'f1': min(f1)},
    #                     {'precision': max(precision), 'recall': max(recall), 'f1': max(f1)})
    people, sketches, mask_ind = _init_heatmaps()
    precision = {'adult': precision[:21], 'child': precision[21:]}
    recall = {'adult': recall[:21], 'child': recall[21:]}
    f1 = {'adult': f1[:21], 'child': f1[21:]}
    for person in people:
        for _set, name in [(precision, 'precision'), (recall, 'recall'), (f1, 'f1')]:
            img_cpy = sketches[person].copy()
            colormap = plt.get_cmap('RdYlGn')
            for region, value in enumerate(_set[person]):
                print(f'{person} - {name}: {value}')
                # x = (value - minimum[name]) / (maximum[name] - minimum[name])
                r, g, b = colormap(value)[:3]
                img_cpy[mask_ind[person] == (region + 1)] = (255 * b, 255 * g, 255 * r)  # for cv2 the order is bgr
            cv2.imwrite(os.path.join(save_dir, f'{person}_{name}.png'), img_cpy)
    #     cv2.imshow(person, img_cpy)
    # cv2.waitKey(0)


def tsne_on_annotations(annots_matrix, n_components=2):
    svd = TruncatedSVD(n_components=50, n_iter=50, random_state=42)
    svd.fit(annots_matrix)
    annots_matrix = svd.transform(annots_matrix)
    for perplexity in [25]:
        print(perplexity)
        t_sne = manifold.TSNE(
            n_components=n_components,
            perplexity=perplexity,
            init="pca",
            n_iter=250,
            random_state=0
        )
        S_t_sne = t_sne.fit_transform(annots_matrix)
    return S_t_sne


def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


def plot_2d(points, points_color, metadata, title):
    fig = px.scatter(
        x=points[:, 0],
        y=points[:, 1],
        color=[colors[p_color] for p_color in points_color],
        title=title,
        hover_data=[metadata]
    )
    fig.show()


def plot_2d_matplotlib(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()


colors = ["#696969", "#808080", "#a9a9a9", "#c0c0c0", "#dcdcdc", "#2f4f4f", "#556b2f", "#8b4513", "#6b8e23", "#a0522d",
          "#a52a2a", "#2e8b57", "#228b22", "#7f0000", "#191970", "#006400", "#708090", "#808000", "#483d8b", "#b22222",
          "#5f9ea0", "#3cb371", "#bc8f8f", "#663399", "#b8860b", "#bdb76b", "#008b8b", "#cd853f", "#4682b4", "#d2691e",
          "#9acd32", "#20b2aa", "#cd5c5c", "#00008b", "#4b0082", "#32cd32", "#daa520", "#8fbc8f", "#8b008b", "#b03060",
          "#d2b48c", "#48d1cc", "#66cdaa", "#9932cc", "#ff0000", "#ff4500", "#ff8c00", "#ffa500", "#ffd700", "#6a5acd",
          "#ffff00", "#c71585", "#0000cd", "#40e0d0", "#7fff00", "#00ff00", "#9400d3", "#ba55d3", "#00fa9a", "#8a2be2",
          "#00ff7f", "#4169e1", "#e9967a", "#dc143c", "#00ffff", "#00bfff", "#f4a460", "#9370db", "#0000ff", "#a020f0",
          "#f08080", "#adff2f", "#ff6347", "#da70d6", "#d8bfd8", "#b0c4de", "#ff7f50", "#ff00ff", "#1e90ff", "#db7093",
          "#f0e68c", "#fa8072", "#eee8aa", "#ffff54", "#6495ed", "#dda0dd", "#90ee90", "#add8e6", "#87ceeb", "#ff1493",
          "#7b68ee", "#ffa07a", "#afeeee", "#87cefa", "#7fffd4", "#ffdead", "#ff69b4", "#ffe4c4", "#ffc0cb"]
my_cmap = ListedColormap(colors)


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, cmap=my_cmap, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


def read_data(annot_file_path):
    logging.info(f'Reading data and annotations from: {annot_file_path}')
    with open(annot_file_path, 'r') as f:
        data = json.load(f)
    return data


def convert_annots_to_matrix(annots):
    # this method converts annotations into (N, 21*21) matrix and subject number as labels
    no_of_frames = len([frame for subject in annots for frame in annots[subject]])
    annots_matrix = np.zeros((no_of_frames, 21*21))
    labels = np.zeros(no_of_frames, dtype=int)
    metadata = ['' for _ in range(no_of_frames)]
    count = 0
    for s, subject in enumerate(annots):
        for frame in annots[subject]:
            cur_annot = np.zeros((21, 21))
            for item in annots[subject][frame]:
                cur_annot[item['adult'], item['child']] = 1
            annots_matrix[count, :] = cur_annot.reshape((21*21))
            labels[count] = s
            metadata[count] = f'assets/{subject}/cam1/{frame}'
            count += 1
    return annots_matrix, labels, metadata


def main():
    annots = read_data("data/youth/signature/all/all_signature.json")
    annots_matrix, labels, metadata = convert_annots_to_matrix(annots)
    S_t_sne = tsne_on_annotations(annots_matrix)
    plot_2d(S_t_sne, labels, metadata, "T-distributed Stochastic  \n Neighbor Embedding")


if __name__ == '__main__':
    main()
