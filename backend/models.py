from sentence_transformers import SentenceTransformer
from simpletransformers.classification import ClassificationModel

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
from scipy.special import softmax

import logging


class Model:
    _use_cuda = False
    _DEVICE = 'cpu'

    def __init__(self, name, path_to_weights, cuda=False) -> None:
        if cuda:
            self._use_cuda = True
            self._DEVICE = 'cuda'

        self.model_for_embeddings = SentenceTransformer(name, device=self._DEVICE)
        self.classifier = ClassificationModel('bert', path_to_weights, use_cuda=self._use_cuda)

        logging.warning(f'\n Model device: {self._DEVICE}')

    def process_data(self, data):
        """
        data: list of sentences

        Compute embeddings and write to self filed - embeddings\n
        Compute cosine similarity btw all <count> sentences and write it to self field - cosine_matrix
        
        return pd.DataFrame with columns: ['text', 'category']
        """

        logging.warning(f'\n Model for embeddings: <MiniLM-L12-v2> \n Model for classification: <trained BERT>')

        data = pd.DataFrame({
            'text': data,
            'category': np.zeros(len(data)),
            'unique': np.ones(len(data))
        })

        data['text'] = data['text'].astype('str').apply(self.clean_text)
        logging.warning('\n Model: text cleaned')

        categorized_data_frame = self.make_classification(data)

        detected_data = self.detect_duplicate(categorized_data_frame)

        proc_data = self.delete_duplicate(detected_data)

        proc_data['category'] = proc_data['category'].apply(self.id2cat)

        proc_data.drop(['unique'], axis=1, inplace=True)

        logging.warning('\n Model: index turned to name')
        logging.warning('\n Model: Its all done')

        return proc_data

    def make_classification(self, data):
        
        logging.warning('\n Classification...')
        predict_classes, raw_outputs = self.classifier.predict(data['text'].values.tolist())

        proba = softmax(raw_outputs, axis=1)

        max_proba = np.partition(proba, -1)[:, -1]
        predict_classes[max_proba < 0.4] = -1

        mask_1 = data['text'].str.contains('купил')
        mask_2 = data['text'].str.contains('продал')
        mask = mask_1 + mask_2
        mask = ~mask
        
        # data['proba'] = max_proba
        data['category'] = predict_classes*mask
        
        
        logging.warning(f'\n len predicted: {len(predict_classes)}, unique: {np.unique(predict_classes)}')
        logging.warning('\n Model: categorized done')

        return data

    def detect_duplicate(self, data, t=0.78):
        # индексы, которые удалим
        all_indexes = []

        for category in data['category'].unique():
            temp_data = data[data['category'] == category].reset_index()
            list_of_sentences = temp_data['text'].tolist()

            # получаем эмбединги текстов
            embeddings = self.model_for_embeddings.encode(list_of_sentences)

            # считаем косинусное расстояние между текстами
            cosine = cosine_similarity(embeddings)

            # оставляем только значения выше порога
            cosine = np.where(cosine < t, False, True)

            # индексы, которые потом удалим
            indexes_to_drop = list()

            # выбираем эти индексы
            for i in range(len(cosine)):
                temp_indexes = np.transpose((cosine[i][i + 1:]).nonzero())
                indexes_to_drop.extend(list(temp_indexes + 1))

            # получаем индексы полного датафрейма, а не нашего среза (где только 1 категория новостей)
            true_indexes = (np.where(np.isin(temp_data.index.values, indexes_to_drop), temp_data['index'], -1))

            # удалим дубликаты
            true_indexes = set(true_indexes)

            # удалим -1 - (индекс не является дубликатом)
            true_indexes.remove(-1)

            # добавим во все итоговые индексы
            all_indexes.extend(true_indexes)

        # не будем удалять защищенные индексы
        predicted_indexes = set(self.get_trades(data))
        indexes_to_drop = list(set(all_indexes) - set(predicted_indexes))

        # дропаем
        data.drop(indexes_to_drop, inplace=True)
        data.reset_index(inplace=True, drop=True)

        return data

    def get_trades(self, data):
        contains_kupil = data['text'].str.contains('🟢купил')
        contains_prodal = data['text'].str.contains('⭕️продал')

        k_indexes = contains_kupil.where(contains_kupil == True).dropna().index.values.tolist()
        p_indexes = contains_prodal.where(contains_prodal == True).dropna().index.values.tolist()

        # возвращает индексы
        return k_indexes + p_indexes

    def make_embeddings(self, data):
        embds = self.model_for_embeddings.encode(data)
        return embds

    def delete_duplicate(self, data):
        data = data[data['unique'] == 1]
        logging.warning('\n Model: duplicates are deleted')
        return data

    def clean_text(self, text):
        text = str(text)
        return str(re.sub("[\(\[].*?[\)\]]", "", text))

    def id2cat(self, i: int) -> str:
        categories = ['Экономика', 'Технологии', 'Политика', 'Шоубиз', 'Криптовалюты', 'Путешествия',
                      'Образование и познавательное', 'Развлечения и юмор', 'Новости и СМИ',
                      'Психология', 'Искусство', 'Спорт', 'Цитаты', 'Еда и кулинария', 'Блоги', 'Бизнес и стартапы',
                      'Маркетинг, PR, реклама', 'Дизайн', 'Право',
                      'Мода и красота', 'Здоровье и медицина', 'Софт и приложения', 'Видео и фильмы', 'Музыка', 'Игры',
                      'Рукоделие', 'Другое']
        name = categories[i]
        return name
