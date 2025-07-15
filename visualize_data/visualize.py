import os
import cv2
import pandas as pd
import collections
import ffmpeg
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize ground truth labels for a video')
    parser.add_argument('--video_id', type=str, default=' ', help='Video ID to visualize')
    parser.add_argument('--output_dir', type=str, default=' ', help='Output directory')
    parser.add_argument('--video_dir', type=str, default=' ', help='Directory containing videos')
    parser.add_argument('--audio_dir', type=str, default=' ', help='Directory containing audio files')
    parser.add_argument('--csv_path', type=str, default=' ', help='Path to ground truth CSV file')
    parser.add_argument('--name', type=str, default=' ', help='Name of the video')
    
    return parser.parse_args()

def add_audio(video_path, audio_path, output_path): # code from: https://stackoverflow.com/questions/56973205/how-to-combine-the-video-and-audio-files-in-ffmpeg-python
    input_video = ffmpeg.input(video_path)
    input_audio = ffmpeg.input(audio_path)
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output_path).run()


def main():
    args = parse_args()
    video_dir = args.video_dir
    csv_path = args.csv_path
    output_dir = args.output_dir
    audio_dir = args.audio_dir
    name = args.name
    
    output_video = os.path.join(output_dir, f'{name}.mp4')

    # Choose a video_id to visualize (change as needed)
    video_id = args.video_id
    video_file = os.path.join(video_dir, f"{video_id}.mp4")
    audio_file = os.path.join(audio_dir, f"{video_id}.wav")

    gt_df = pd.read_csv(csv_path)
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
        if name == 'ground_truth':
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
        elif name == 'model_results':
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
    
    # Add audio to the annotated video (not the original video)
    add_audio(output_video, audio_file, os.path.join(output_dir, f'{name}_with_audio.mp4'))
    print(f'Video with audio saved as {os.path.join(output_dir, f"{name}_with_audio.mp4")}')

    os.remove(output_video) # remove video without audio

if __name__ == '__main__':
    main()