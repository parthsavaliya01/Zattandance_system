import cv2

video_path = "/home/admin1/attendance_system/recordings/2026-08-21/camera111.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Let's search around video seconds 8200 to 8280 for 09:50:00
print("Searching for 09:50:00...")
start_frame = int(8200 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
for f in range(start_frame, start_frame + 2000, 25): # check every 1 second
    ret, frame = cap.read()
    if not ret:
        break
    # We can print the OSD text using a simple OCR, or let's save a frame every 5 seconds to inspect
    sec = f / fps
    if (f - start_frame) % 125 == 0: # every 5 seconds
        cv2.imwrite(f"/home/admin1/attendance_system/outputs/osd_check/search_start_{int(sec)}.jpg", frame[0:100, 0:800])
    cap.set(cv2.CAP_PROP_POS_FRAMES, f + 25)

# Let's search around video seconds 9160 to 9240 for 10:10:00
print("Searching for 10:10:00...")
end_frame = int(9160 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
for f in range(end_frame, end_frame + 2000, 25):
    ret, frame = cap.read()
    if not ret:
        break
    sec = f / fps
    if (f - end_frame) % 125 == 0: # every 5 seconds
        cv2.imwrite(f"/home/admin1/attendance_system/outputs/osd_check/search_end_{int(sec)}.jpg", frame[0:100, 0:800])
    cap.set(cv2.CAP_PROP_POS_FRAMES, f + 25)

cap.release()
print("Search done!")
