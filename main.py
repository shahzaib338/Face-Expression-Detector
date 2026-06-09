import torch.nn as nn
import torch
from torchvision import models, transforms
from PIL import Image
import cv2


model = models.resnet18()
model.fc = nn.Linear(512,7)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

model.load_state_dict(torch.load("fer13_resnet18_70.pth", map_location='cpu'))
model.eval()
print("Model Loaded Successfuly")

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera Error")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            _,pred = torch.max(output, 1)
            confidence = torch.softmax(output,1)
            emotion = classes[pred.item()]
            conf_score = confidence[0][pred].item()

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
        label = f"{emotion} {conf_score *100:.2f}%"
        cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.imshow("FaceDetector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("Camera Closes")

     