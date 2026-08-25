import cv2
import os

video_path = "/home/admin1/attendance_system/recordings/2026-08-21/camera111.mp4"
output_dir = "/home/admin1/attendance_system/outputs/osd_check"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"FPS: {fps}")
print(f"Total Frames: {total_frames}")
print(f"Duration: {duration} seconds ({duration/3600:.2f} hours)")

# We will sample frames every 30 minutes (1800 seconds)
# 1800 seconds * 25 fps = 45000 frames
for hour in range(0, 11):
    for half in [0, 30]:
        minutes = hour * 60 + half
        seconds = minutes * 60
        frame_idx = int(seconds * fps)
        if frame_idx >= total_frames:
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Crop the top-right / top-left area where the OSD timestamp usually is (typically top 10% of the image)
            h, w = frame.shape[:2]
            # Save the whole frame or a cropped area containing the OSD. Let's save a cropped top part to save space, say top 150 pixels.
            # Usually cameras show OSD on top-right or top-left.
            crop = frame[0:150, :]
            filename = f"frame_{hour:02d}_{half:02d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, crop)
            print(f"Saved {filename} at {seconds} seconds (frame {frame_idx})")

cap.release()
print("Done!")
