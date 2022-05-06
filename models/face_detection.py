import cv2 
import os

RESULT_PATH = '/content/results'      # The path that the result images will be saved
LOG_PATH = '/content/log.txt'         # The path for the working log file
RESIZE = (128,128)                

vid = cv2.VideoCapture(vid_path)       # Read video
# Parse into frames 
frame_buffer = []               # A list to hold frame images
frame_buffer_color = []         # A list to hold original frame images
success = 1
while(success):
  success, frame = vid.read()
  if not success: break
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  frame_buffer.append(gray)
  # frame_buffer_color.append(frame)
vid.release()

face_detector = dlib.get_frontal_face_detector()   

face_buffer = []
for (i, image) in enumerate(frame_buffer):
  face_rects = face_detector(image,1)
  if len(face_rects) < 1:
    print("No face detected: ",vid_path)
    logfile.write(vid_path + " : No face detected \r\n")
    break
    if len(face_rects) > 1:
      print("Too many face: ",vid_path)
      logfile.write(vid_path + " : Too many face detected \r\n")
      break
    rect = face_rects[0]
    face_buffer.append(rect)

cropped_buffer = []
for (i,landmark) in enumerate(face_buffer):
  cropped = frame_buffer[i][landmark[2]:landmark[3],landmark[0]:lanedmark[1]]  
  cropped = cv2.resize(cropped,(RESIZE[0],RESIZE[1]),interpolation=cv2.INTER_CUBIC)       
  cropped_buffer.append(cropped)
  
vid_name = "test_vid"

directory = RESULT_PATH + vid_name + "/"
for (i,image) in enumerate(cropped_buffer):
  if not os.path.exists(directory):
    os.makedirs(directory)
  cv2.imwrite(directory + "%d"%(i+1) + ".jpg", image)     # Write face image
