import cv2
import time
import os
import argparse
from dotenv import load_dotenv

# Load local environment variables
load_dotenv(override=True)

# Default to environment variables, then sensible fallbacks
DEFAULT_USERNAME = os.getenv("CAMERA_USERNAME", "admin")
DEFAULT_PASSWORD = os.getenv("CAMERA_PASSWORD", "")
DEFAULT_IP = os.getenv("CAMERA_IP_1", "127.0.0.1")

def main():
    parser = argparse.ArgumentParser(description="Record RTSP stream from camera to an MP4 file.")
    parser.add_argument("--ip", type=str, default=DEFAULT_IP, help="Camera IP address")
    parser.add_argument("--username", type=str, default=DEFAULT_USERNAME, help="Camera RTSP Username")
    parser.add_argument("--password", type=str, default=DEFAULT_PASSWORD, help="Camera RTSP Password")
    parser.add_argument("--duration", type=int, default=1800, help="Recording duration in seconds (default 1800s / 30m)")
    parser.add_argument("--output-dir", type=str, default="test_output", help="Directory to save output video")
    
    args = parser.parse_args()

    rtsp_url = f"rtsp://{args.username}:{args.password}@{args.ip}:554/Streaming/Channels/101"
    
    print(f"🎥 Connecting to RTSP Stream on {args.ip}...")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("❌ Unable to connect to camera. Check IP, credentials, or RTSP URL route.")
        exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    print(f"📡 Connection established:")
    print(f"   - FPS: {fps}")
    print(f"   - Resolution: {width} x {height}")
    print(f"   - Codec: mp4v")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"output_video_{args.ip}.mp4")
    out = cv2.VideoWriter(output_file, fourcc, fps if fps > 0 else 25, (width, height))

    start_time = time.time()
    print(f"⏺️ Recording stream to '{output_file}' for {args.duration} seconds...")

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_count += 1
            out.write(frame)

            if time.time() - start_time > args.duration:
                print("⏹️ Duration reached. Stopping recording.")
                break
    except KeyboardInterrupt:
        print("\n⏹️ Recording interrupted by user. Stopping.")
    finally:
        cap.release()
        out.release()

    print(f"✅ Saved output video as: {output_file}")

if __name__ == "__main__":
    main()