import face_recognition
import cv2
import os
import glob
import numpy as np

class FaceRecognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.5

    def load_encoding_images(self, images_path):
        # Load Images
        #biblioteca glob serve pra buscar arquivos de um diretorio
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("Codificando {} imagens encontradas.".format(len(images_path)))

        # Armazenando os nomes e a codificação das imagens (cada img_path pro images_path (que contem as 6 imagens)
        for img_path in images_path:
            #imread() carrega uma imagem dos arquivos do for
            img = cv2.imread(img_path)
            # converte o formato de cores da imagem de BGR (formato como ela foi codificada - que o OpenCV usa),
            # para RGB (formato como ela precisa estar para ser reconecida pelo face_recognition)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Obtém o nome do arquivo + a extensão
            basename = os.path.basename(img_path)

            # o ext do formato splitext busca o onme do arquivo até o .extensãodoarquivo (ou seja, só o  nome do arquivo)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding: Dada uma imagem, retorne a codificação de face de 128 dimensões para cada face na imagem.
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Armazenando os arquivos de codifcações e os nomes das imagens detro do known_face_encodings e known_face_names respectiv.
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Nomes das imagens e respectivas codificações carregadas")

    #chamo essa função no main passando, onde o frame passado pra ela é o que a câmera ta capturando
    def detect_known_faces(self, frame):
        # redimensionar o tamanho das imagens recebidas
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)

        # Encontra todos os rostos e codificações de rosto no quadro atual do vídeo
        # Converte as imagens de BGR para RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Retorna uma matriz de caixas delimitadoras de rostos humanos em uma imagem
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # não consegui printar os pontos na imagem, mas as coordenadas x,y consigo no print
        face_landmarks = face_recognition.face_landmarks(rgb_small_frame)

        # Dada uma imagem dentro da caixa delimitadora, retorna a codificação de face de 128 dimensões para cada face na imagem
        # i.e. um vetor com 128 valores
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        # pra cada codificação de face dentro do face_encodings: (ou seja, cada rosto que aparece na tela da cam)
        for face_encoding in face_encodings:
            # Compara as faces codificadas com os rostos codificados conhecidos (known_face_encodings)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Desconhecido"
            # Setar inicialmente o nome que aparece junto a tela da câmera = Desconhecido

            # Se encontra uma correspondencia em known_face_encodings, já utiliza essa primeira
            # Se corresponde com algu [True]:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Ou, em vez disso, use a face conhecida com a menor distância até a nova face
            # o face_distance retorna um array com as distancias de cada face conhecida (kown..) para aquela(s) que tão na cam
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            #passo o array com as distancias na função argmin <- que retorna o índice do mínimo valor desse array
            best_match_index = np.argmin(face_distances)
            # se ele encontra algum índice (ou seja, tinha pelo menos uma face com caixa delimitadora na cam):
            if matches[best_match_index]:
                # o 'name' agora recebe o nome daquela pessoa (buscada pelo índice em knnown_face_names)
                name = self.known_face_names[best_match_index]
            # e adiciono ao array face_names [porque pode ter mais de uma face na cam]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        #retorna a localização para as caixas delimitadoras, e os nomes encontrados de cada face
        return face_locations.astype(int), face_names, face_landmarks
