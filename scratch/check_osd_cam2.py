import cv2
import os

video_path = "/home/admin1/attendance_system/recordings/2026-08-22/camera11112.mp4"
output_dir = "/home/admin1/attendance_system/outputs/osd_check_cam2"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Total Frames: {total_frames}")

# Check OSD times at specific video seconds
check_seconds = [0, 18000, 25200, 28800, 32400, 34200, 36000, 37800, 39600]

for seconds in check_seconds:
    frame_idx = int(seconds * fps)
    if frame_idx >= total_frames:
        break
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        filename = f"frame_{seconds}.jpg"
        filepath = os.path.join(output_dir, filename)
        # Crop the top OSD bar (height: 150px)
        cv2.imwrite(filepath, frame[0:150, :])
        print(f"Saved {filename} at {seconds} seconds (frame {frame_idx})")

cap.release()
print("Done!")
