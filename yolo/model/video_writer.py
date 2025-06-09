import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, Counter
import argparse
import os
from pathlib import Path

class VideoObjectDetector:
    def __init__(self, model_path, confidence_threshold=0.5, iou_threshold=0.45):
        """
        Initialize the video object detector

        Args:
            model_path (str): Path to YOLO model file
            confidence_threshold (float): Minimum confidence for detections
            iou_threshold (float): IoU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_counts = Counter()
        self.total_frames = 0

    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on frame

        Args:
            frame: Input frame
            results: YOLO detection results

        Returns:
            tuple: (frame with drawn detections, frame_counts dict)
        """
        frame_counts = Counter()

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Get confidence and class
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]

                    # Filter by confidence
                    if confidence >= self.confidence_threshold:
                        # Count detection for this frame
                        frame_counts[class_name] += 1

                        # Update total counts (for summary)
                        self.class_counts[class_name] += 1

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        ## Draw label with confidence
                        #label = f"{class_name}: {confidence:.2f}"
                        #label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                        ## Draw label background
                        #cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                        #            (x1 + label_size[0], y1), (0, 255, 0), -1)

                        ## Draw label text
                        #cv2.putText(frame, label, (x1, y1 - 5),
                        #          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame, frame_counts

    def add_frame_counts_overlay(self, frame, frame_counts):
        """
        Add per-frame detection counts to the top right corner

        Args:
            frame: Input frame
            frame_counts: Counter object with current frame's detection counts

        Returns:
            frame: Frame with per-frame counts overlay
        """
        if not frame_counts:
            return frame

        h, w = frame.shape[:2]

        # Calculate text dimensions for positioning
        y_offset = 30
        max_width = 0

        # First pass to calculate overlay size
        texts = []
        for class_name, count in sorted(frame_counts.items()):
            text = f"{class_name.title()}: {count}"
            texts.append(text)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            max_width = max(max_width, text_size[0])

        if texts:
            # Create semi-transparent background
            overlay = frame.copy()
            overlay_height = len(texts) * 30 + 20
            overlay_x = w - max_width - 30
            cv2.rectangle(overlay, (overlay_x - 10, 10),
                         (w - 10, overlay_height), (0, 0, 0), -1)

            # Blend overlay with frame
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Add text for each class count
            for text in texts:
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = w - text_size[0] - 20

                cv2.putText(frame, text, (text_x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30

        return frame

    def process_video(self, input_path, output_path, show_preview=False):
        """
        Process video with object detection

        Args:
            input_path (str): Path to input video
            output_path (str): Path to output video
            show_preview (bool): Whether to show real-time preview
        """
        # Open video capture
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {input_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Reset counters
        self.class_counts.clear()
        self.total_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.total_frames += 1

                # Run YOLO inference
                results = self.model(frame, conf=self.confidence_threshold,
                                   iou=self.iou_threshold, verbose=False)

                # Draw detections and get frame counts
                frame, frame_counts = self.draw_detections(frame, results)

                # Add per-frame counts overlay (top right corner)
                frame = self.add_frame_counts_overlay(frame, frame_counts)

                # Write frame to output video
                out.write(frame)

                # Show preview if requested
                if show_preview:
                    cv2.imshow('Object Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Print progress
                if self.total_frames % 30 == 0:
                    progress = (self.total_frames / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({self.total_frames}/{total_frames})")

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")

        finally:
            # Clean up
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        # Print completion message
        print(f"\nProcessing complete!")
        print(f"Output saved to: {output_path}")
        print(f"Total frames processed: {self.total_frames}")



def main():
    parser = argparse.ArgumentParser(description='Process video with YOLO object detection')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('output_video', help='Path to output video file')
    parser.add_argument('--model', default='yolov8n.pt',
                       help='Path to YOLO model file (default: yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--preview', action='store_true',
                       help='Show real-time preview window')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}")
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Initialize detector
        detector = VideoObjectDetector(
            model_path=args.model,
            confidence_threshold=args.conf,
            iou_threshold=args.iou
        )

        # Process video
        detector.process_video(
            input_path=args.input_video,
            output_path=args.output_video,
            show_preview=args.preview
        )

    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()
