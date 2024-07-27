import cv2
import pyaudio
from ultralytics import YOLO
import wave

#wavを再生
def play_wav(filename):
    wav_file = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wav_file.getsampwidth()),
                    channels=wav_file.getnchannels(),
                    rate=wav_file.getframerate(),
                    output=True)

    data = wav_file.readframes(1024)
    while data:
        stream.write(data)
        data = wav_file.readframes(1024)

    stream.stop_stream()
    stream.close()
    p.terminate()

# Load the YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

# Open the web camera stream
cap = cv2.VideoCapture(0)

border = 225


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    k = cv2.waitKey(1)
    if k != -1:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy[0].tolist()
        # Assuming that keypoints[5] is left hip, keypoints[6] is right shoulder, etc.
        nose_x = int(keypoints[0][0])
        nose_y = int(keypoints[0][1])
        print(f"鼻x={nose_x},鼻y={nose_y}")
        
    if nose_y > border:
        play_wav('sounds/badposition.wav')
        print("姿勢悪いぞ!!")
    cv2.line(annotated_frame,(0,border),(800,border),(0,255,0),3)
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
