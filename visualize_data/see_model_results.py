import os, cv2, collections
import pandas as pd


def main():
    # --- Overlay bounding boxes on real video frames ---
    # Find the first video file in orig_videos
    video_dir = '../AVDIAR2ASD/orig_videos'
    video_id = 'Seq21-2P-S1M1'
    video_file = os.path.join(video_dir, f"{video_id}.mp4")
    bbox_df = pd.read_csv('../outputs/0_res.csv')
    df = bbox_df[bbox_df['video_id'] == video_id]

    # Get the video_id from the first row of the CSV
    video_id = bbox_df.iloc[0]['video_id']
    video_file = os.path.join(video_dir, f"{video_id}.mp4")
    print('Using video:', video_file)

    cap = cv2.VideoCapture(video_file)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {FPS}")

    # Print first 10 frame_timestamps from CSV and what frame index they would correspond to
    print("First 10 CSV timestamps and their expected frame indices:")
    for ts in df['frame_timestamp'].head(10):
        print(f"timestamp: {ts} -> expected frame idx: {int(round(float(ts) * FPS))}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('outputs/model_results.mp4', fourcc, FPS, (W, H))

    # Build mapping from frame_idx to list of rows (boxes)
    frame_to_boxes = collections.defaultdict(list)
    for _, row in df.iterrows():
        frame_idx = int(round(float(row['frame_timestamp']) * FPS))
        frame_to_boxes[frame_idx].append(row)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
            pred = 1 if row['score'] >= 0.5 else 0
            color = (0, 255, 0) if pred == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
            cv2.putText(frame, f'Score: {row["score"]:.2f}', (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        out.write(frame)

    cap.release()
    out.release()
    print('Video with real frames and boxes saved as output_with_boxes.mp4')

    # --- Play the generated video using OpenCV ---
    video_path = 'output_with_boxes.mp4'
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video Playback', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to quit early
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()