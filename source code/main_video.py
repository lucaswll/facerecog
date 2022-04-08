import cv2
from face_recog import FaceRecognition
from datetime import datetime

# instancializa a classe face_recog()
sfr = FaceRecognition()

dataIniCodif = datetime.now()
#timeIni = dataIni.strftime('%H:%M:%S:%f')
minIniCodif = dataIniCodif.strftime('%M')
secIniCodif = dataIniCodif.strftime('%S')
mSecIniCodif = dataIniCodif.strftime('%f')
print('Min:Sec:mSec da codificação: ', minIniCodif, secIniCodif, mSecIniCodif)
#pra calcular o tempo gasto desde inicio da codificação das imagens, até reconhecimento final

#usa a função de load encoding, encontra e codifica as imagens desta pasta
sfr.load_encoding_images("imagens3/")

# Inicia a captura de vídeo (0 pra identificar minha camera (unica) do pc)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    dataIni = datetime.now()
    # timeIni = dataIni.strftime('%H:%M:%S:%f')
    minIni = dataIni.strftime('%M')
    secIni = dataIni.strftime('%S')
    mSecIni = dataIni.strftime('%f')
    print('Min:Sec:mSec inicial detecção', minIni, secIni, mSecIni) #pra calcular o tempo de reconhecimento [assim que realiza a detecção]

    # Detect Faces
    # Chamo a função de detectar as faces, passando o frame da cam como param
    # E vinculo o retorno as var 'face_locations' e 'face_names'
    face_locations, face_names, face_landmarks = sfr.detect_known_faces(frame)

    #print(face_landmarks)

    # função zip no python retorna uma lista de tuplas [estrutura de dados tipo uma lista, mas imutável]
    # face_loc <- a localização das faces (face_locat) || name <- face_names

    for face_loc, name in zip(face_locations, face_names):
        # as 4 coordenadas retornadas pela caixa delimitadora
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        #traz a escrita dos nomes e a imagem do retangulo
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_ITALIC, 1, (200, 200, 200), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)

        print('Nome: ', name)

        dataEnd = datetime.now()
        # timeIni = dataIni.strftime('%H:%M:%S:%f')
        minEnd = dataEnd.strftime('%M')
        secEnd = dataEnd.strftime('%S')
        mSecEnd = dataEnd.strftime('%f')
        print('Min:Sec:mSec final reconhecimento', minEnd, secEnd, mSecEnd) #tempo quando reconhece

    cv2.imshow("Frame", frame)

    #espera 5 ms após digitar a tecla ESC (27) para parar o while
    key = cv2.waitKey(5)
    if key == 27:
        break

# desativa a webcam
cap.release()

# fecha a janela aberta (sem apresentar erro)
cv2.destroyAllWindows()