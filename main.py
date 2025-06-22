import telebot
import onnxruntime as ort
import numpy as np
import cv2
import os
import logging

# Базовое логгирование для отладки
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best.onnx')

# Список названий классов (должен соответствовать порядку в модели)
CLASS_NAMES = ['Dent', 'Scratch', 'Crack', 'Glass shatter', 'Lamp broken', 'Tire flat']

# Цвета, которые используются для отрисовки bounding box'ов для каждого класса повреждений
COLORS = {
    'Dent': (0, 255, 0),
    'Scratch': (255, 0, 0),
    'Crack': (0, 0, 255),
    'Glass shatter': (255, 255, 0),
    'Lamp broken': (0, 255, 255),
    'Tire flat': (255, 0, 255)
}

TARGET_SIZE = (640, 640) # Размер изображения для инференса модели

CONFIDENCE_THRESHOLD = 0.25 # Порог для детекции
NMS_THRESHOLD = 0.45 # Порог для NMS

# Предобрабатывает входное изображение для подачи в модель
def preprocess(image, target_size=TARGET_SIZE):
    height, width = image.shape[:2]

    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    img = rgb.astype(np.float32) / 255.0

    img = img.transpose((2, 0, 1)) # HWC -> CHW

    img = np.expand_dims(img, axis=0).astype(np.float32) # Добавление измерения дял батча

    return img, (width / target_size[0], height / target_size[1])


# Получает выходы модели и извлекает информацию о bounding box'ах
def bboxes_info(outputs, class_names, threshold=CONFIDENCE_THRESHOLD):
    proposals = outputs[0][0]

    bboxes = []

    # Находим максимальную оценку среди всех классов для каждого предложенного bounding box'а
    # и фильтруем их по порогу уверенности
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


# Отрисовывает bounding box'ы на изображении после применения NMS и возвращает список отфильтрованных детекций.
def draw_and_filter_detections(image, bboxes, scales, colors, score_threshold=CONFIDENCE_THRESHOLD, nms_threshold=NMS_THRESHOLD):
    img = image.copy()

    coords_list = []
    scores_list = []
    labels_list = []

    for bbox in bboxes:
        xc, yc, w, h = bbox['coords']

        # Преобразовываем центральные координаты в угловые
        x1 = int((xc - w / 2) * scales[0])
        y1 = int((yc - h / 2) * scales[1])
        x2 = int((xc + w / 2) * scales[0])
        y2 = int((yc + h / 2) * scales[1])

        coords_list.append([x1, y1, x2, y2])
        scores_list.append(bbox['score'])
        labels_list.append(bbox['class'])

    ids = cv2.dnn.NMSBoxes(coords_list, scores_list, score_threshold, nms_threshold) # Применяем NMS
    
    filtered_detections = []

    if len(ids) == 0:
        pass
        
    else:
        coords = [coords_list[i] for i in ids]
        labels = [labels_list[i] for i in ids]
        scores = [scores_list[i] for i in ids]

        for (x1, y1, x2, y2), label, score in zip(coords, labels, scores):
            color = colors.get(label, (0, 0, 0))

            # Вычисляем размер текста для правильного расположения фона
            text = f'{label}: {score:.2f}'
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2) # Рамка вокруг объекьта
            cv2.rectangle(image, (x1 - 1, y1 - text_h - 10), (x1 + text_w, y1), color, -1) # Закрашенный прямоугольник для фона текста
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # Текст

            filtered_detections.append({'class': label, 'score': score})

    return image, filtered_detections

def main():
    
    TOKEN = os.getenv('TOKEN')
    if not TOKEN:
        logging.error("Токен для телеграм бота не найден.")
        exit(1)
    
    bot = telebot.TeleBot(TOKEN)

    try:
        session = ort.InferenceSession(MODEL_PATH)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        logging.info("Модель успешно загружена.")
    except Exception as e:
        logging.error(f"Ошибка при загрузке модели: {e}")
        exit(1)

    # Обработчик команды /start
    @bot.message_handler(commands=['start'])
    def start(message):
        bot.send_message(message.chat.id, 'Привет! Отправь фотографию машины, чтобы узнать, есть ли повреждения.')
    
    # Обработчик входящих фотографий
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
            
            # Предобработка изображения
            input_tensor, scales = preprocess(image)
            
            # Выполнение инференса модели
            outputs = session.run([output_name], {input_name: input_tensor})
            
            # Получение информации о bounding box'ах
            bboxes = bboxes_info(outputs, CLASS_NAMES)
            
            # Применение NMS и отрисовка детектированных объектов
            result, detections = draw_and_filter_detections(image, bboxes, scales, COLORS)
            
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

if __name__ == "__main__":
    main()