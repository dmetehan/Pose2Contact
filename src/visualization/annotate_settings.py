# we annotate a part of the data in terms of meaningful "interaction settings" such as
# picking up
# sitting on lap
# supporting
# touching
# holding (arm/hands/legs)
# child holding parent
# And then do post-hoc analyses with these
# labels to see how, semantically, we're doing.
import cv2
import os.path
import numpy as np
from visualize import read_data, convert_annots_to_matrix


def wait_label():
    while True:
        key = cv2.waitKey(0)
        if key == ord("q") or key == ord("Q"):
            return -1
        elif key in [ord("p"), ord("P")]:
            return "p"  # picking up
        elif key in [ord("l"), ord("L")]:
            return "l"  # sitting on lap
        elif key in [ord("s"), ord("S")]:
            return "s"  # supporting
        elif key in [ord("t"), ord("T")]:
            return "t"  # touching
        elif key in [ord("h"), ord("H")]:
            return "h"  # holding (arm/hands/legs)
        elif key in [ord("c"), ord("C")]:
            return "c"  # child holding parent


already_annotated = set()
image_dir = "assets"
test_samples_info = read_data("../../workdir/youth/signature/temp/save_preds.json")  # all_preds, all_labels, metadata
output_file = "interaction_settings.txt"
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        for line in f:
            subject, frame, _ = line.split(",")
            already_annotated.add((subject.strip(), frame.strip()))
for meta in test_samples_info['metadata']:
    subject, frame = meta[0][0], meta[1][0]
    if (subject, frame) in already_annotated:
        continue
    img1_path = os.path.join(image_dir, subject, f"cam1", frame)
    img2_path = os.path.join(image_dir, subject, f"cam2", frame)
    img3_path = os.path.join(image_dir, subject, f"cam3", frame)
    img4_path = os.path.join(image_dir, subject, f"cam4", frame)
    img1 = np.concatenate((cv2.imread(img1_path), cv2.imread(img2_path)), axis=1)
    img2 = np.concatenate((cv2.imread(img3_path), cv2.imread(img4_path)), axis=1)
    img = np.concatenate((img1, img2), axis=0)
    win_name = f"{subject}, {frame}"
    cv2.imshow(win_name, img)
    cv2.moveWindow(win_name, 10, 10)
    label = wait_label()
    cv2.destroyAllWindows()
    if label == -1:
        break
    with open(output_file, "a+") as f:
        f.write(f"{subject}, {frame}, {label}\n")
    already_annotated.add((subject, frame))
