"""
Simple Popup Labeler
Label popups by drawing bounding boxes.

Usage:
    python label_popups.py --images /path/to/images --labels /path/to/labels

Controls:
    Click + Drag: Draw box
    s: Save and next
    d: Skip
    u: Undo last box
    q: Quit
"""

import cv2
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    label_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        images.extend(list(image_dir.glob(f"*{ext}")))
        images.extend(list(image_dir.glob(f"*{ext.upper()}")))
    images = sorted(set(images))

    print(f"Found {len(images)} images")
    print("Controls: s=save/next, d=skip, u=undo, q=quit")

    idx = 0
    boxes = []
    current_box = None
    orig_img = None
    scale = 1.0

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_box, boxes
        if event == cv2.EVENT_LBUTTONDOWN:
            x0, y0 = int(x / scale), int(y / scale)
            current_box = [x0, y0, x0, y0]
        elif event == cv2.EVENT_MOUSEMOVE and current_box:
            current_box[2] = int(x / scale)
            current_box[3] = int(y / scale)
        elif event == cv2.EVENT_LBUTTONUP and current_box:
            x1, y1 = current_box[2], current_box[3]
            x0, y0 = current_box[0], current_box[1]
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            if x1 - x0 > 5 and y1 - y0 > 5:
                img_h, img_w = orig_img.shape[:2]
                xc = (x0 + x1) / 2 / img_w
                yc = (y0 + y1) / 2 / img_h
                bw = (x1 - x0) / img_w
                bh = (y1 - y0) / img_h
                boxes.append((0, xc, yc, bw, bh))
            current_box = None

    cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Labeler", 800, 600)
    cv2.setMouseCallback("Labeler", mouse_callback)

    while idx < len(images):
        img_path = images[idx]
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            idx += 1
            continue

        img_h, img_w = orig_img.shape[:2]

        # Always scale to fit window
        win_w, win_h = 790, 590
        scale = min(win_w / img_w, win_h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)

        # Load existing
        boxes = []
        label_file = label_dir / f"{img_path.stem}.txt"
        if label_file.exists():
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, xc, yc, w, h = int(parts[0]), *map(float, parts[1:5])
                        boxes.append((cls, xc, yc, w, h))

        current_box = None

        while True:
            disp = cv2.resize(orig_img, (disp_w, disp_h))

            # Draw boxes
            for cls, xc, yc, bw, bh in boxes:
                x1 = int((xc - bw / 2) * img_w * scale)
                y1 = int((yc - bh / 2) * img_h * scale)
                x2 = int((xc + bw / 2) * img_w * scale)
                y2 = int((yc + bh / 2) * img_h * scale)
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if current_box:
                x1 = int(current_box[0] * scale)
                y1 = int(current_box[1] * scale)
                x2 = int(current_box[2] * scale)
                y2 = int(current_box[3] * scale)
                cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.putText(
                disp,
                f"{idx + 1}/{len(images)}: {img_path.name[:40]}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                disp,
                f"Boxes: {len(boxes)} | s=save d=skip u=undo q=quit",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )

            cv2.imshow("Labeler", disp)
            key = cv2.waitKey(0) & 0xFF

            if key == ord("s"):
                with open(label_file, "w") as f:
                    for cls, xc, yc, bw, bh in boxes:
                        f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                print(f"Saved: {img_path.name} ({len(boxes)} boxes)")
                idx += 1
                break
            elif key == ord("d"):
                print(f"Skipped: {img_path.name}")
                idx += 1
                break
            elif key == ord("u") and boxes:
                boxes.pop()
                print(f"Undo: {len(boxes)} boxes")
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
