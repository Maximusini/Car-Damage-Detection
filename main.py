import telebot
import onnxruntime as ort
import numpy as np
import cv2
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess(image, target_size=(640, 640)):
    height, width = image.shape[:2]

    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    img = rgb.astype(np.float32) / 255.0

    img = img.transpose((2, 0, 1)) # HWC -> CHW

    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img, (width / target_size[0], height / target_size[1])

def bboxes_info(outputs, class_names, threshold=0.25):
    proposals = outputs[0][0]

    bboxes = []

    max_scores = np.max(proposals[4:], axis=0)
    bbox_ids = np.where(max_scores >= threshold)[0]

    for i in bbox_ids:
        data = proposals[:, i]
        coords = data[:4]
        scores = data[4:]

        max_score = np.max(scores)
        class_id = np.argmax(scores)
        label = class_names[class_id]

        bboxes.append({
            'coords': coords,
            'class': label,
            'score': float(max_score)
        })

    return bboxes

def draw_and_filter_detections(image, bboxes, scales, colors, class_names):
    img = image.copy()

    coords_list = []
    scores_list = []
    labels_list = []

    for bbox in bboxes:
        xc, yc, w, h = bbox['coords']

        x1 = int((xc - w / 2) * scales[0])
        y1 = int((yc - h / 2) * scales[1])
        x2 = int((xc + w / 2) * scales[0])
        y2 = int((yc + h / 2) * scales[1])

        coords_list.append([x1, y1, x2, y2])
        scores_list.append(bbox['score'])
        labels_list.append(bbox['class'])

    ids = cv2.dnn.NMSBoxes(coords_list, scores_list, score_threshold=0.25, nms_threshold=0.45)
    
    filtered_detections = []

    if len(ids) == 0:
        pass
        
    else:
        coords = [coords_list[i] for i in ids]
        labels = [labels_list[i] for i in ids]
        scores = [scores_list[i] for i in ids]

        for (x1, y1, x2, y2), label, score in zip(coords, labels, scores):
            color = colors.get(label, 0)

            text = f'{label}: {score:.2f}'
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(image, (x1 - 1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            filtered_detections.append({'class': label, 'score': score})

    return image, filtered_detections


if __name__ == "__main__":
    
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best.onnx')

    COLORS = {
        'Dent': (0, 255, 0),
        'Scratch': (255, 0, 0),
        'Crack': (0, 0, 255),
        'Glass shatter': (255, 255, 0),
        'Lamp broken': (0, 255, 255),
        'Tire flat': (255, 0, 255)
    }

    CLASS_NAMES = ['Dent', 'Scratch', 'Crack', 'Glass shatter', 'Lamp broken', 'Tire flat']
    
    TOKEN = os.getenv('TOKEN')

    bot = telebot.TeleBot(TOKEN)

    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    @bot.message_handler(commands=['start'])
    def start(message):
        bot.send_message(message.chat.id, 'Привет! Отправь фотографию машины, чтобы узнать есть ли повреждения.')

    @bot.message_handler(content_types=['photo'])
    def handle_photo(message):
        bot.send_message(message.chat.id, 'Анализирую фото...')

        photo_path = ''
        result_path = ''
        
        try:
            fileID = message.photo[-1].file_id
            file_info = bot.get_file(fileID)

            downloaded_file = bot.download_file(file_info.file_path)

            photo_path = f'{fileID}.jpg'
            
            with open(photo_path, 'wb') as f:
                f.write(downloaded_file)
            
            image = cv2.imread(photo_path)
            input_tensor, scales = preprocess(image)
            
            outputs = session.run([output_name], {input_name: input_tensor})
            
            bboxes = bboxes_info(outputs, CLASS_NAMES)
            
            result, detections = draw_and_filter_detections(image, bboxes, scales, COLORS, CLASS_NAMES)
            
            if not detections:
                bot.send_message(message.chat.id, "Повреждения не обнаружены!")
                
            else:
                result_path = f'{fileID}_result.jpg'
                cv2.imwrite(result_path, result)
                
                with open(result_path, 'rb') as f:
                    bot.send_photo(message.chat.id, f)
                
                
                summary = "Обнаруженные повреждения:\n"
                for detection in detections:
                    summary += f"- {detection['class']}: {detection['score']:.2f}\n"
                
                bot.send_message(message.chat.id, f"{summary} Всего: {len(detections)}")
                
            
        except Exception as e:
            bot.send_message(message.chat.id, 'Ошибка при обработке фото.')
            logging.error(f"Ошибка при обработке фото: {e}")
            
        finally:
            if photo_path and os.path.exists(photo_path):
                os.remove(photo_path)
            if result_path and os.path.exists(result_path):
                os.remove(result_path)


    bot.polling(none_stop=True, interval=0)