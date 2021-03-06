import dlib
import cv2
import os

!wget -nd http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
!bunzip2 /content/shape_predictor_68_face_landmarks.dat.bz2

RESULT_PATH = '/content/results'      # The path that the result images will be saved
LOG_PATH = '/content/log.txt'         # The path for the working log file
LIP_MARGIN = 0.3                # Marginal rate for lip-only image.
RESIZE = (64,64)                # Final image size
logfile = open(LOG_PATH,'w')
# Face detector and landmark detector
face_detector = dlib.get_frontal_face_detector()   
landmark_detector = dlib.shape_predictor("/content/shape_predictor_68_face_landmarks.dat")	# Landmark detector path

def shape_to_list(shape):
	coords = []
	for i in range(0, 68):
	  coords.append((shape.part(i).x, shape.part(i).y))
	return coords

vid_path = "/content/video.mp4"

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

landmark_buffer = []
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
    landmark = landmark_detector(image, rect)   # Detect face landmarks
    landmark = shape_to_list(landmark)
    landmark_buffer.append(landmark)

cropped_buffer = []
for (i,landmark) in enumerate(landmark_buffer):
  lip_landmark = landmark[48:68]                                          
  lip_x = sorted(lip_landmark,key = lambda pointx: pointx[0])             
  lip_y = sorted(lip_landmark, key = lambda pointy: pointy[1])
  x_add = int((-lip_x[0][0]+lip_x[-1][0])*LIP_MARGIN)                     
  y_add = int((-lip_y[0][1]+lip_y[-1][1])*LIP_MARGIN)
  crop_pos = (lip_x[0][0]-x_add, lip_x[-1][0]+x_add, lip_y[0][1]-y_add, lip_y[-1][1]+y_add)   
  cropped = frame_buffer[i][crop_pos[2]:crop_pos[3],crop_pos[0]:crop_pos[1]]        
  cropped = cv2.resize(cropped,(RESIZE[0],RESIZE[1]),interpolation=cv2.INTER_CUBIC)       
  cropped_buffer.append(cropped)

vid_name = "test_vid"

directory = RESULT_PATH + vid_name + "/"
for (i,image) in enumerate(cropped_buffer):
  if not os.path.exists(directory):
    os.makedirs(directory)
  cv2.imwrite(directory + "%d"%(i+1) + ".jpg", image)     # Write lip image

logfile.close()

