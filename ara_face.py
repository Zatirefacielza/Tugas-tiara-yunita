import cv2

face_ref = cv2.CascadeClassifier("ref_face_ara.xml")
camera = cv2.VideoCapture(0)

def face_detection(frame):
    print(f"Processing frame with shape {frame.shape}")
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    print(f"Found {len(faces)} faces")
    return faces

def drawer_box(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame")
            break
        faces = face_detection(frame)
        drawer_box(frame, faces)
        cv2.imshow("ARA FACE", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    close_window()

if __name__ == "__main__":
    main()
