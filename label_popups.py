"""
Simple Popup Labeler
==================
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


class Labeler:
    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.label_dir.mkdir(parents=True, exist_ok=True)

        # Get images
        self.images = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            self.images.extend(list(self.image_dir.glob(f"*{ext}")))
            self.images.extend(list(self.image_dir.glob(f"*{ext.upper()}")))
        self.images = sorted(set(self.images))

        self.current_idx = 0
        self.boxes = []
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_box = None

    def load_boxes(self, img_path):
        """Load existing boxes"""
        label_file = self.label_dir / f"{img_path.stem}.txt"
        boxes = []
        if label_file.exists():
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        boxes.append((cls, xc, yc, w, h))
        return boxes

    def save_boxes(self, img_path, boxes):
        """Save boxes in YOLO format"""
        if not boxes:
            label_file = self.label_dir / f"{img_path.stem}.txt"
            if label_file.exists():
                label_file.unlink()
            return

        img = cv2.imread(str(img_path))
        if img is None:
            return
        h, w = img.shape[:2]

        label_file = self.label_dir / f"{img_path.stem}.txt"
        with open(label_file, "w") as f:
            for cls, xc, yc, bw, bh in boxes:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    def draw(self, img, boxes, current_box):
        """Draw boxes on image"""
        disp = img.copy()

        # Draw existing boxes
        for cls, xc, yc, bw, bh in boxes:
            x1 = int((xc - bw / 2) * img.shape[1])
            y1 = int((yc - bh / 2) * img.shape[0])
            x2 = int((xc + bw / 2) * img.shape[1])
            y2 = int((yc + bh / 2) * img.shape[0])
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw current box
        if current_box:
            x1, y1, x2, y2 = current_box
            cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return disp

    def mouse(self, event, x, y, flags, param):
        """Mouse callback"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x = x
            self.start_y = y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = (self.start_x, self.start_y, x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.current_box:
                x1, y1, x2, y2 = self.current_box
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                if x2 - x1 > 10 and y2 - y1 > 10:
                    # Convert to YOLO format
                    img_h, img_w = self.current_img.shape[:2]
                    xc = (x1 + x2) / 2 / img_w
                    yc = (y1 + y2) / 2 / img_h
                    bw = (x2 - x1) / img_w
                    bh = (y2 - y1) / img_h
                    self.boxes.append((0, xc, yc, bw, bh))

                self.current_box = None

    def run(self):
        """Main loop"""
        cv2.namedWindow("Labeler")
        cv2.moveWindow("Labeler", 100, 100)

        total = len(self.images)

        while self.current_idx < total:
            img_path = self.images[self.current_idx]
            self.current_img = cv2.imread(str(img_path))

            if self.current_img is None:
                self.current_idx += 1
                continue

            cv2.setMouseCallback("Labeler", self.mouse)
            self.boxes = self.load_boxes(img_path)
            self.current_box = None

            showing = True
            while showing:
                disp = self.draw(self.current_img, self.boxes, self.current_box)

                cv2.putText(
                    disp,
                    f"{self.current_idx + 1}/{total}: {img_path.name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    disp,
                    f"Boxes: {len(self.boxes)} | s=save/next d=skip u=undo q=quit",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

                cv2.imshow("Labeler", disp)
                key = cv2.waitKey(100) & 0xFF

                if key == ord("s"):
                    self.save_boxes(img_path, self.boxes)
                    print(f"Saved: {img_path.name} ({len(self.boxes)} boxes)")
                    self.current_idx += 1
                    showing = False

                elif key == ord("d"):
                    print(f"Skipped: {img_path.name}")
                    self.current_idx += 1
                    showing = False

                elif key == ord("u"):
                    if self.boxes:
                        self.boxes.pop()
                        print(f"Undo: {len(self.boxes)} boxes")

                elif key == ord("q"):
                    cv2.destroyAllWindows()
                    return

        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    labeler = Labeler(args.images, args.labels)
    print(f"Found {len(labeler.images)} images")
    print("Controls: s=save/next, d=skip, u=undo, q=quit")
    labeler.run()
