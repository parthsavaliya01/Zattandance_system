import cv2

video_path = "/home/admin1/attendance_system/recordings/2026-08-22/camera11112.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Let's search around video seconds 30300 to 30380 for 17:30:00
print("Searching for 17:30:00...")
start_frame = int(30300 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
for f in range(start_frame, start_frame + 2000, 25): # check every 1 second
    ret, frame = cap.read()
    if not ret:
        break
    sec = f / fps
    if (f - start_frame) % 125 == 0: # every 5 seconds
        cv2.imwrite(f"/home/admin1/attendance_system/outputs/osd_check_cam2/search_start_{int(sec)}.jpg", frame[0:150, 0:800])
    cap.set(cv2.CAP_PROP_POS_FRAMES, f + 25)

# Let's search around video seconds 31020 to 31100 for 17:45:00
print("Searching for 17:45:00...")
end_frame = int(31020 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
for f in range(end_frame, end_frame + 2000, 25):
    ret, frame = cap.read()
    if not ret:
        break
    sec = f / fps
    if (f - end_frame) % 125 == 0: # every 5 seconds
        cv2.imwrite(f"/home/admin1/attendance_system/outputs/osd_check_cam2/search_end_{int(sec)}.jpg", frame[0:150, 0:800])
    cap.set(cv2.CAP_PROP_POS_FRAMES, f + 25)

cap.release()
print("Search done!")
