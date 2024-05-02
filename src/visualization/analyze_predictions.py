import json
import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score
import visualize
from matplotlib.ticker import AutoMinorLocator


def evaluate(gts, preds, title, print_keys=False):
    all_keys = ['6x6', '21x21']
    if print_keys:
        print("Model", end='\t&\t')
        for k, key in enumerate(all_keys):
            print(f"{key}", end='\t&\t' if k < len(all_keys) - 1 else '\t\\\\\n')
    print(title, end='\t&\t')
    for k, key in enumerate(all_keys):
        print(f"{jaccard_score(gts[key], preds[key], average='micro'):.3f}", end='\t&\t' if k < len(all_keys) - 1 else '\t\\\\')
    print()


def average(x, y):
    return (x + y) / 2


def fuse_scores(scores1, scores2, fn):
    thresholds = {'12': 0.25, '6x6': 0.125, '42': 0.2, '21x21': 0.0625}
    fused = {task: [[1 if fn(scores1[task][s][i], scores2[task][s][i]) > thresholds[task] else 0 for i, _ in enumerate(sample)] for s, sample in enumerate(scores1[task])] for task in scores1}
    return fused


def remove_no_contact_cases(results):
    results = {key: {task.replace('*', 'x'): results[key][task] for task in results[key]} if isinstance(results[key], dict) else results[key] for key in results}
    all_keys = [key for key in results]
    new_results = {key: {task: [] for task in results[key]} if isinstance(results[key], dict) else [] for key in results}
    for task in results['labels']:
        for e, elem in enumerate(results['labels'][task]):
            if sum(elem) > 0:
                for key in all_keys:
                    if isinstance(results[key], dict):
                        new_results[key][task].append(results[key][task][e])
                    else:
                        new_results[key].append(results[key][e])
    return new_results


def box_and_whiskers_test_set():
    with open("src/visualization/save_preds_gcn_adaptive.json") as f:
        gcn_results = json.load(f)
    with open("src/visualization/save_preds_resnet.json") as f:
        resnet_results = json.load(f)
    resnet_results = remove_no_contact_cases(resnet_results)
    print(len(resnet_results['labels']['21x21']))
    assert resnet_results['labels'] == gcn_results['labels']
    evaluate(gcn_results['labels'], gcn_results['preds'], 'GCN', print_keys=True)
    evaluate(resnet_results['labels'], resnet_results['preds'], 'ResNet')
    for fn, name in zip([average, min, max], ['Average', 'Min', 'Max']):
        fused_preds = fuse_scores(gcn_results['scores'], resnet_results['scores'], fn)
        evaluate(resnet_results['labels'], fused_preds, f'{name} Fusion')
        if name == 'Min':
            save_dir = 'src/visualization'
            kwargs = {'average': 'micro'}
            visualize.vis_box_and_whiskers_per_setting_score(resnet_results['labels'], fused_preds, resnet_results['metadata'], jaccard_score, save_dir, **kwargs)


def draw_timeline(preds12):
    preds_12 = np.array(preds12).T
    # Create a DataFrame with all NaN values initially
    plot_data = pd.DataFrame(preds_12)

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 2))
    cax = ax.matshow(plot_data, cmap='Blues', aspect='auto', interpolation='none')

    # Set up axes
    if len(preds12[0]) == 12:
        ax.set_yticks(np.arange(12), ["parent head", "parent core", "parent larm", "parent rarm", "parent lleg", "parent rleg",
                                      "infant head", "infant core", "infant larm", "infant rarm", "infant lleg", "infant rleg"])
    else:
        ax.set_yticks(np.arange(8), ["parent head", "parent core", "parent arms", "parent legs", "infant head", "infant core", "infant arms", "infant legs"])
    # ax.set_xticks(np.arange(len(preds_12)))
    ax.set_xlabel('Seconds')
    ax.set_title('Predictions per Time Frame and Body Part')

    # Add a color bar to indicate the markers
    # plt.colorbar(cax, ax=ax, orientation='vertical', label='Prediction (1 = Predicted)')
    minor_locator = AutoMinorLocator(2)
    plt.gca().yaxis.set_minor_locator(minor_locator)
    plt.grid(axis='y', which='minor')
    # ax.grid(axis='y')
    plt.show()


def combine_preds(binary, segmentation, pose_detections):
    combined = []
    for i, sample in enumerate(segmentation):
        if binary[i] == 0 or len(pose_detections['preds'][str(i)]) < 2:
            combined.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            # if binary[i] != 0 and len(pose_detections['preds'][str(i)]) < 2:
            #     print('yey')
        else:
            if sum(sample[:6]) == 0 or sum(sample[6:]) == 0:
                combined.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            else:
                combined.append(sample)
    return combined


def test_video(root_dir, threshold):
    with open(os.path.join(root_dir, "test_video_contact_binary.txt")) as f:
        contact_no_cont_gt = json.load(f)
    with open(os.path.join(root_dir, "test_video_preds.json")) as f:
        test_results = json.load(f)
    with open(os.path.join(root_dir, "pose_detections.json")) as f:
        pose_detections = json.load(f)
    max_threshold = 0.9
    preds12 = np.array(test_results['scores']['12'])
    # for threshold in [0.25, 0.35, 0.45, 0.55, 0.75]:
    cur_thresh_preds = np.logical_and(preds12 > threshold, preds12 < max_threshold)
    # cur_thresh_preds = preds12 > threshold
    # draw_timeline(cur_thresh_preds)
    final_preds = combine_preds(contact_no_cont_gt, cur_thresh_preds, pose_detections)
    return final_preds


def combine_arms_legs(preds):
    combined = []
    for pred12 in preds:
        combined.append([pred12[0], pred12[1], (pred12[2] | pred12[3]), (pred12[4] | pred12[5]),
                         pred12[6], pred12[7], (pred12[8] | pred12[9]), (pred12[10] | pred12[11])])
    return combined


def gen_teaser_image():
    processed1 = test_video("src/visualization/test_videos/B50284", 0.3)
    processed2 = test_video("src/visualization/test_videos/B00432", 0.3)  # 0.25
    final1 = combine_arms_legs(processed1)
    final2 = combine_arms_legs(processed2)
    draw_timeline(final1)
    draw_timeline(final2)


def main():
    box_and_whiskers_test_set()
    # gen_teaser_image()


if __name__ == '__main__':
    main()
