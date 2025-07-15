import os
import cv2
import pandas as pd
import collections

video_dir = '../AVDIAR2ASD/orig_videos'
gt_csv = '../AVDIAR2ASD/csv/val_labels.csv'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
output_video = os.path.join(output_dir, 'ground_truth.mp4')

# Choose a video_id to visualize (change as needed)
video_id = 'Seq21-2P-S1M1'
video_file = os.path.join(video_dir, f"{video_id}.mp4")

gt_df = pd.read_csv(gt_csv)
df = gt_df[gt_df['video_id'] == video_id]
if len(df) > 0:
    print("Example box row:", df.iloc[0])
else:
    print(f"No boxes found for video_id: {video_id}")
    exit()

cap = cv2.VideoCapture(video_file)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS)
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, FPS, (W, H))

# Build mapping from frame_idx to list of rows (boxes)
frame_to_boxes = collections.defaultdict(list)
for _, row in df.iterrows():
    frame_idx = int(round(float(row['frame_timestamp']) * FPS))
    frame_to_boxes[frame_idx].append(row)
    
print(f"Total boxes: {len(df)}")
for frame_idx in sorted(frame_to_boxes.keys())[:20]:  # first 20 frames with boxes
    print(f"Frame {frame_idx}: {len(frame_to_boxes[frame_idx])} boxes")
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video has {num_frames} frames")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start

for i in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break
    # Draw all boxes for this frame
    for row in frame_to_boxes.get(i, []):
        x1 = int(row['entity_box_x1'] * W)
        y1 = int(row['entity_box_y1'] * H)
        x2 = int(row['entity_box_x2'] * W)
        y2 = int(row['entity_box_y2'] * H)
        # Green for SPEAKING_AUDIBLE, red for NOT_SPEAKING
        color = (0, 255, 0) if row['label'] == 'SPEAKING_AUDIBLE' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
        cv2.putText(frame, row['label'], (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    out.write(frame)

cap.release()
out.release()
print(f'Video with ground truth boxes saved as {output_video}') 