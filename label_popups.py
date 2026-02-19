"""
Simple popup labeler using mouse clicks
Draw boxes around popups - click and drag

Usage:
    python label_popups.py --images /path/to/images --labels /path/to/labels

Controls:
    Click + Drag: Draw box
    s: Save and next
    d: Skip (don't save)
    a: Previous
    c: Clear all boxes
    q: Quit
"""

import cv2
import argparse
from pathlib import Path


class SimpleLabeler:
    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.label_dir.mkdir(parents=True, exist_ok=True)

        # Get all image types
        self.images = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            self.images.extend(list(self.image_dir.glob(f"*{ext}")))
            self.images.extend(list(self.image_dir.glob(f"*{ext.upper()}")))
        self.images = sorted(set(self.images))

        self.current_idx = 0
        self.boxes = []
        self.drawing = False
        self.start_x = self.start_y = 0
        self.current_box = None
        self.img_h = 0
        self.img_w = 0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = (self.start_x, self.start_y, x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.start_x, x), min(self.start_y, y)
            x2, y2 = max(self.start_x, x), max(self.start_y, y)
            if x2 - x1 > 10 and y2 - y1 > 10:
                self.boxes.append((x1, y1, x2, y2))
            self.current_box = None

    def load_existing_labels(self):
        label_file = self.label_dir / f"{self.images[self.current_idx].stem}.txt"
        boxes = []
        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, xc, yc, w, h = map(float, parts[:5])
                        x1 = int((xc - w / 2) * self.img_w)
                        y1 = int((yc - h / 2) * self.img_h)
                        x2 = int((xc + w / 2) * self.img_w)
                        y2 = int((yc + h / 2) * self.img_h)
                        boxes.append((x1, y1, x2, y2))
        return boxes

    def save_labels(self):
        if not self.boxes:
            # Remove label file if no boxes
            label_file = self.label_dir / f"{self.images[self.current_idx].stem}.txt"
            if label_file.exists():
                label_file.unlink()
            return
        label_file = self.label_dir / f"{self.images[self.current_idx].stem}.txt"
        with open(label_file, "w") as f:
            for x1, y1, x2, y2 in self.boxes:
                xc = (x1 + x2) / 2 / self.img_w
                yc = (y1 + y2) / 2 / self.img_h
                w = (x2 - x1) / self.img_w
                h = (y2 - y1) / self.img_h
                f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    def run(self):
        cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Labeler", self.mouse_callback)

        while self.current_idx < len(self.images):
            img = cv2.imread(str(self.images[self.current_idx]))
            if img is None:
                self.current_idx += 1
                continue

            self.img_h, self.img_w = img.shape[:2]
            display = img  # Full native resolution
            scale = 1.0

            self.boxes = self.load_existing_labels()

            while True:
                disp = display.copy()

                # Draw existing boxes
                for x1, y1, x2, y2 in self.boxes:
                    sx1, sy1 = int(x1 * scale), int(y1 * scale)
                    sx2, sy2 = int(x2 * scale), int(y2 * scale)
                    cv2.rectangle(disp, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)

                # Draw current box
                if self.current_box:
                    x1, y1, x2, y2 = self.current_box
                    sx1, sy1 = int(x1 * scale), int(y1 * scale)
                    sx2, sy2 = int(x2 * scale), int(y2 * scale)
                    cv2.rectangle(disp, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)

                # Info
                cv2.putText(
                    disp,
                    f"Image {self.current_idx + 1}/{len(self.images)}: {self.images[self.current_idx].name[:40]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    disp,
                    f"Boxes: {len(self.boxes)} | s=save/next d=skip a=prev c=clear q=quit",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

                cv2.imshow("Labeler", disp)
                key = cv2.waitKey(0) & 0xFF

                if key == ord("q"):
                    cv2.destroyAllWindows()
                    return
                elif key == ord("s"):
                    self.save_labels()
                    print(
                        f"Saved: {self.images[self.current_idx].name} ({len(self.boxes)} boxes)"
                    )
                    self.current_idx += 1
                    break
                elif key == ord("d"):
                    print(f"Skipped: {self.images[self.current_idx].name}")
                    self.current_idx += 1
                    break
                elif key == ord("a"):
                    self.current_idx = max(0, self.current_idx - 1)
                    break
                elif key == ord("c"):
                    self.boxes = []
                    print("Cleared all boxes")

        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    labeler = SimpleLabeler(args.images, args.labels)
    print(f"Found {len(labeler.images)} images")
    print("Controls: s=save/next, d=skip, a=prev, c=clear, q=quit")
    labeler.run()
