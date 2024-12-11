import cv2
from ultralytics import YOLO

# 1. Carregar o modelo YOLO
model = YOLO('face_detection_2.pt')  # Substitua por yolov8s.pt ou o caminho para o seu modelo personalizado

# 2. Inicializar a câmera (ID 0 é geralmente a câmera padrão do computador)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a câmera!")
    exit()

# 3. Configurar as dimensões do frame e o FPS da câmera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 4. Processar a câmera frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame da câmera!")
        break

    # Fazer a predição usando o YOLO
    results = model.predict(frame, conf=0.5)  # Ajuste o limite de confiança conforme necessário

    # Obter o frame com as detecções plotadas
    annotated_frame = results[0].plot()  # Adiciona caixas delimitadoras e rótulos no frame

    # Mostrar o frame processado
    cv2.imshow('YOLO Prediction - Camera', annotated_frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Liberar os recursos
cap.release()
cv2.destroyAllWindows()

print("Processamento da câmera concluído.")
