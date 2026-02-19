"""
Simple Popup Labeler
===================
Label popups in screenshots by drawing bounding boxes.

Usage:
    python label_popups.py --images /path/to/images --labels /path/to/labels

Controls:
    Mouse Click + Drag: Draw box
    's': Save and next image
    'd': Skip image
    'q': Quit
    'u': Undo last box
"""

import cv2
import os
import sys
import argparse
from pathlib import Path
import glob


class SimpleLabeler:
    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.label_dir.mkdir(parents=True, exist_ok=True)

        # Get all images
        extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        self.images = []
        for ext in extensions:
            self.images.extend(sorted(self.image_dir.glob(f"*{ext}")))
            self.images.extend(sorted(self.image_dir.glob(f"*{ext.upper()}")))

        # Remove duplicates
        self.images = sorted(list(set(self.images)))

        self.current_idx = 0
        self.boxes = []  # Current image boxes
        self.all_boxes = {}  # All saved boxes

        # Mouse state
        self.drawing = False
        self.start_x = self.start_y = 0
        self.current_box = None

        # Window
        self.window_name = "Popup Labeler"

    def load_image_boxes(self, image_path):
        """Load existing boxes if label file exists"""
        label_file = self.label_dir / f"{image_path.stem}.txt"
        boxes = []

        if label_file.exists():
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO format: class x_center y_center width height
                        cls = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])

                        # Convert to pixel coords
                        img = cv2.imread(str(image_path))
                        if img is not None:
                            h_img, w_img = img.shape[:2]
                            x1 = int((xc - w / 2) * w_img)
                            y1 = int((yc - h / 2) * h_img)
                            x2 = int((xc + w / 2) * w_img)
                            y2 = int((yc + h / 2) * h_img)
                            boxes.append((x1, y1, x2, y2))

        return boxes

    def save_boxes(self, image_path):
        """Save boxes in YOLO format"""
        if not self.boxes:
            # Remove label file if no boxes
            label_file = self.label_dir / f"{image_path.stem}.txt"
            if label_file.exists():
                label_file.unlink()
            return

        img = cv2.imread(str(image_path))
        if img is None:
            return

        h, w = img.shape[:2]

        label_file = self.label_dir / f"{image_path.stem}.txt"

        with open(label_file, "w") as f:
            for x1, y1, x2, y2 in self.boxes:
                # Convert to YOLO format
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        self.all_boxes[str(image_path)] = self.boxes.copy()

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
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
                # Normalize coordinates
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # Only add if box is big enough
                if x2 - x1 > 10 and y2 - y1 > 10:
                    self.boxes.append((x1, y1, x2, y2))
                self.current_box = None
                self.current_box = None

    def draw_boxes(self, img):
        """Draw all boxes on image"""
        # Draw saved boxes
        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{i + 1}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Draw current box
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return img

    def run(self):
        """Main loop"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        total = len(self.images)
        labeled = 0

        while self.current_idx < total:
            img_path = self.images[self.current_idx]

            # Load existing boxes
            self.boxes = self.load_image_boxes(img_path)

            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                self.current_idx += 1
                continue

            # Create display image (resized if too big)
            display = img.copy()
            h, w = display.shape[:2]
            max_dim = 1200
            if h > max_dim or w > max_dim:
                scale = max_dim / max(h, w)
                display = cv2.resize(display, (int(w * scale), int(h * scale)))

            orig_h, orig_w = h, w
            scale = orig_w / w if display.shape[1] != w else 1

            while True:
                disp = self.draw_boxes(display.copy())

                # Info overlay
                info = f"Image {self.current_idx + 1}/{total} | Boxes: {len(self.boxes)} | {img_path.name}"
                cv2.putText(
                    disp,
                    info,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    disp,
                    "Controls: s=save/next, d=skip, u=undo, q=quit",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

                cv2.imshow(self.window_name, disp)

                key = cv2.waitKey(10) & 0xFF

                if key == ord("s"):
                    # Save and next
                    if scale != 1:
                        # Scale boxes back
                        self.boxes = [
                            (
                                int(x1 * scale),
                                int(y1 * scale),
                                int(x2 * scale),
                                int(y2 * scale),
                            )
                            for x1, y1, x2, y2 in self.boxes
                        ]
                    self.save_boxes(img_path)
                    labeled += 1
                    print(f"Saved: {img_path.name} ({len(self.boxes)} boxes)")
                    break

                elif key == ord("d"):
                    # Skip
                    print(f"Skipped: {img_path.name}")
                    break

                elif key == ord("u"):
                    # Undo last box
                    if self.boxes:
                        self.boxes.pop()
                        print(f"Undo: {len(self.boxes)} boxes remaining")

                elif key == ord("q"):
                    # Quit
                    cv2.destroyAllWindows()
                    print(f"\nLabeled {labeled} images")
                    return

        cv2.destroyAllWindows()
        print(f"\nDone! Labeled {labeled}/{total} images")


def main():
    parser = argparse.ArgumentParser(description="Simple Popup Labeler")
    parser.add_argument("--images", required=True, help="Folder containing images")
    parser.add_argument("--labels", required=True, help="Folder to save labels")

    args = parser.parse_args()

    labeler = SimpleLabeler(args.images, args.labels)
    print(f"Found {len(labeler.images)} images")
    print(f"Labels will be saved to: {args.labels}")
    print("\nControls:")
    print("  Click + Drag: Draw box")
    print("  's': Save and next image")
    print("  'd': Skip image")
    print("  'u': Undo last box")
    print("  'q': Quit")
    print()

    labeler.run()


if __name__ == "__main__":
    main()
