OUTPUT_DIR="outputs"
VIDEO_DIR="../AVDIAR2ASD/orig_videos"
AUDIO_DIR="../AVDIAR2ASD/orig_audios"
VIDEO_ID="Seq21-2P-S1M1"
CSV_PATH="../AVDIAR2ASD/csv/val_labels.csv"
MODEL_RESULTS_CSV_PATH="../outputs/0_res.csv"


python visualize.py --name ground_truth --video_id $VIDEO_ID --output_dir $OUTPUT_DIR --video_dir $VIDEO_DIR --audio_dir $AUDIO_DIR --csv_path $CSV_PATH 
python visualize.py --name model_results --video_id $VIDEO_ID --output_dir $OUTPUT_DIR --video_dir $VIDEO_DIR --audio_dir $AUDIO_DIR --csv_path $MODEL_RESULTS_CSV_PATH 
