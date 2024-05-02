import os

import cv2


def annotate(root_dir):
    with open(os.path.join(root_dir, "test_video_contact_binary.txt"), "r") as f:
        lines = f.readlines()
        skip_count = len(lines)
    with open(os.path.join(root_dir, "test_video_contact_binary.txt"), "a+") as f:
        for filename in sorted(os.listdir(os.path.join(root_dir, "crops"))):
            if skip_count > 0:
                skip_count -= 1
                continue
            print(filename)
            filepath = os.path.join(root_dir, "crops", filename)
            cv2.imshow("Frame", cv2.imread(filepath))
            key = cv2.waitKey(0)
            if key == ord("c"):
                f.write("1,\n")
            elif key == ord("n"):
                f.write("0,\n")
            elif key == ord("q"):
                break


def main():
    annotate("src/visualization/test_videos/B00432")


if __name__ == '__main__':
    main()
