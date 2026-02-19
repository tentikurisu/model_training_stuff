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
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    label_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    images = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        images.extend(list(image_dir.glob(f"*{ext}")))
        images.extend(list(image_dir.glob(f"*{ext.upper()}")))
    images = sorted(set(images))

    print(f"Found {len(images)} images")
    print("Controls: s=save/next, d=skip, u=undo, q=quit")

    idx = 0
    boxes = []
    drawing = False
    start_x = 0
    start_y = 0
    current_box = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_x, start_y, current_box, img_h, img_w
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_x = x
            start_y = y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_box = (start_x, start_y, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if current_box:
                x1, y1, x2, y2 = current_box
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                if x2 - x1 > 10 and y2 - y1 > 10:
                    xc = (x1 + x2) / 2 / img_w
                    yc = (y1 + y2) / 2 / img_h
                    bw = (x2 - x1) / img_w
                    bh = (y2 - y1) / img_h
                    boxes.append((0, xc, yc, bw, bh))
                current_box = None

    cv2.namedWindow("Labeler")
    cv2.setMouseCallback("Labeler", mouse_callback)

    while idx < len(images):
        img_path = images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue

        img_h, img_w = img.shape[:2]

        # Load existing labels
        boxes = []
        label_file = label_dir / f"{img_path.stem}.txt"
        if label_file.exists():
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        boxes.append((cls, xc, yc, w, h))

        current_box = None

        while True:
            disp = img.copy()

            # Draw existing boxes
            for cls, xc, yc, bw, bh in boxes:
                x1 = int((xc - bw / 2) * img_w)
                y1 = int((yc - bh / 2) * img_h)
                x2 = int((xc + bw / 2) * img_w)
                y2 = int((yc + bh / 2) * img_h)
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw current box
            if current_box:
                x1, y1, x2, y2 = current_box
                cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Info text
            cv2.putText(
                disp,
                f"{idx + 1}/{len(images)}: {img_path.name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                disp,
                f"Boxes: {len(boxes)} | s=save d=skip u=undo q=quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            cv2.imshow("Labeler", disp)

            # Use waitKey(0) to wait indefinitely for key press
            key = cv2.waitKey(0) & 0xFF

            if key == ord("s"):
                # Save
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
            elif key == ord("u"):
                if boxes:
                    boxes.pop()
                    print(f"Undo: {len(boxes)} boxes")
            elif key == ord("q"):
                cv2.destroyAllWindows()
                print("Quit")
                return

    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
