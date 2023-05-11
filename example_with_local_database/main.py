import os

import dj_database_url
import psycopg2
import psycopg2.extras
from psycopg2 import OperationalError

import dedupe
import numpy
from numpy import array

import re
import datetime
import json

import itertools
import time


class Parser():  # класс, который преобразует данные (можно изменить его так, чтобы он мог принимать любые релевантные поля)
    def __init__(self):
        self.dict_of_values = {'phone': None, 'email': None,
                               'created_at': None}
        self.phone_pattern = re.compile(r"(\+7|8|7)[\s-]?(\d{3})[\s-]?(\d{3})[\s-]?(\d{2})[\s-]?(\d{2})")

    def get_values(self, data_json):
        for el in data_json:
            for k, v in el.items():
                if type(v) == list:
                    self.get_values(v)
                else:
                    if type(v) == str:
                        phone_number = re.findall(self.phone_pattern, v)
                        if phone_number:
                            self.dict_of_values['phone'] = "".join(phone_number[0])
                        if '@' in v:
                            self.dict_of_values['email'] = v
                    if type(v) == int:
                        if k == 'created_at':
                            dt_object = datetime.datetime.fromtimestamp(v)
                            formatted_date = dt_object.strftime("%Y-%m-%d %H:%M:%S")
                            self.dict_of_values['created_at'] = formatted_date


def get_input(prompt, valid_values=None, type_=str, min_=None, max_=None, split=False): #функция для проверки корректности ввода
    while True:
        try:
            answer = input(prompt)
            if split:
                answer = [type_(item.strip()) for item in answer.split(', ')]
                for i in answer:
                    if valid_values and i not in valid_values:
                        raise ValueError(f"Неверный ввод. Пожалуйста, введите какие-то из следующих значений: {valid_values}")
            else:
                answer = type_(answer)
                if valid_values and answer not in valid_values:
                    raise ValueError(f"Неверный ввод. Пожалуйста, введите одно из следующих значений: {valid_values}")
                if min_ is not None and answer < min_:
                    raise ValueError(f"Значение не может быть меньше {min_}")
                if max_ is not None and answer > max_:
                    raise ValueError(f"Значение не может быть больше {max_}")
            return answer
        except ValueError as e:
            print(e)


def save_high_confidence_clusters_to_db(clusters, data_for_training): #функция для сохранения дублей в базу данных
    # Cоздаём словарь с кластерами
    cluster_membership = {
        record_id: {"Cluster ID": cluster_id, "confidence_score": score}
        for cluster_id, (records, scores) in enumerate(clusters)
        for record_id, score in zip(records, scores)
    }

    # Создание таблицы id-номер кластера
    id_with_clusters = {
        data_for_training[cluster]['id']: values['Cluster ID']
        for cluster, values in cluster_membership.items()
    }

    # Записываем результаты кластеризации в базу данных
    with write_con.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'base' AND column_name = 'cluster';")
        if not cur.fetchone():
            cur.execute("ALTER TABLE base ADD COLUMN cluster int;")
        write_con.commit()

        for id_value, cluster_value in id_with_clusters.items():
            cur.execute("UPDATE base SET cluster = %s WHERE id = %s", (cluster_value, id_value))
        write_con.commit()

    del cluster_membership, id_with_clusters


def label_uncertain_pairs(deduper, uncertain_pairs):  # функция предлагает проверить сомнительные объекты
    labeled_pairs = []
    for pair in uncertain_pairs:
        print('\n', pair[0], '\n', pair[1])
        label = input('Дубли это или нет? (y/n/u). Если хотите прервать дообучение, введите f: ').lower()

        if label == 'f':
            break
        else:
            labeled_pairs.append({"res": {"__class__": "tuple", "__value__": [pair[0], pair[1]]}, "label": label})

    return labeled_pairs


def get_pairs_for_check(clusters, data_for_training, threshold=0.9): # функция получает пары объектов для проверки
    pairs_for_check = []
    for cluster, scores in clusters:
        counter=0
        for pair, score in zip(itertools.combinations(cluster, 2), scores):
            if score < threshold:
                pairs_for_check.append([data_for_training[pair[0]],data_for_training[pair[1]]])
    return pairs_for_check


def get_high_confident_clusters(clusters): # функция получает уверенные кластеры
    # Создаем список, в который будут добавляться кластеры, у которых для каждого объекта значение принадлежности выше 0.9
    result = []

# Проходим по каждому элементу
    for item in clusters:
        if isinstance(item[0], tuple):
            if all(score >= 0.9 for score in item[1]):
                result.append(item)

    return result


def get_low_confident_clusters(clusters): # функция получает неуверенные кластеры
    # Создаем список, в который будут добавляться кластеры, у которых для каждого объекта значение принадлежности ниже 0.9
    result = []

# Проходим по каждому элементу
    for item in clusters:
        if isinstance(item[0], tuple):
            if all(score < 0.9 for score in item[1]):
                result.append(item)

    return result


def clastering(deduper, data_for_training): # функция для кластеризации и получения новых размеченных данных
    confidence_for_clustering = get_input(
                "Для кластеризации введите нижний порог для определения дублей (прим: Это значение от 0 до 1). Если вы хотите охватить больше возможных дублей и затем их проверить, выбирайте порог ниже. Вы можете экспериментировать: ",
                type_=float,
                min_=0.0,
                max_=1.0
            )
    try:
        clusters = deduper.partition(data_for_training, confidence_for_clustering)
    except dedupe.core.BlockingError as e:
        print("Ошибка при кластеризации:", e)
        print("Попробуйте разметить данные снова, выбрав другие поля или сменив им приоритеты")
        clusters = None
        return False

    print("Кластеризация завершена")

    high_confident_clusters = get_high_confident_clusters(clusters)
    print("Уверенные кластеры получены")
    save_high_confidence_clusters_to_db(high_confident_clusters, data_for_training) # сохранение кластеров с уровнем уверенности выше 0.9 в базу данных
    print(f"Сохранили в базу данных")

    user_input = get_input(
            "Продолжить дообучение? (y/n): ",
            ['y', 'n']
        )
    if user_input.lower() == 'n':
            # запишем веса модели в файл. Сможем потом благодаря нему использовать модель без обучения
        with open(settings_file, 'wb') as sf:
            deduper.write_settings(sf)
        return False
    else:
        low_confident_clusters = get_low_confident_clusters(clusters)
        pairs_for_check = get_pairs_for_check(low_confident_clusters, data_for_training)
        if pairs_for_check:
            labeled = label_uncertain_pairs(deduper, pairs_for_check)
            # тут мы обновляем training file в соответствии с новыми знаниями
            with open(training_file, "r") as file:
                data = json.load(file)

            for i in labeled:
                if i["label"] == "y":
                    data["match"].append(i["res"])
                elif i["label"] == "n":
                    data["distinct"].append(i["res"])

                # Сохранение изменений в файле JSON
            with open(training_file, "w") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
            print("Поздравляем, вы обновили тренировочные данные")
            answer = get_input(
            "Обновить базу данных новыми кластерами? (y/n): ",
            ['y', 'n'])
            if answer.lower() == 'y':
                del clusters
                clastering(deduper, data_for_training)
            else:
                return False
        else:
            print("Дублей не найдено. Вы можете понизить порог кластеризации и попробовать снова")
            answer = get_input(
            "Попробовать снова? (y/n): ",
            ['y', 'n'])
            if answer.lower() == 'y':
                del clusters
                clastering(deduper, data_for_training)
            else:
                return False


if __name__ == '__main__':
    os.environ['DATABASE_URL'] = 'postgresql://user_name:password@localhost:port/db_name'  # задаем адрес для подключения к бд
    db_conf = dj_database_url.config()  # строки для настройки соединения с базой данных

    if not db_conf:
       raise Exception(
           'set DATABASE_URL environment variable with your connection, e.g. '
           'export DATABASE_URL=postgres://user:password@host/mydatabase'
       )

    # ниже задаются курсоры для получения данных из базы данных и записи
    read_con = None
    write_con = None
    while not read_con or not write_con:
        if not read_con:
            try:
                read_con = psycopg2.connect(database=db_conf['NAME'],
                                           user=db_conf['USER'],
                                           password=db_conf['PASSWORD'],
                                           host=db_conf['HOST'],
                                           cursor_factory=psycopg2.extras.RealDictCursor)
                print("Read connection to PostgreSQL DB successful")
            except OperationalError as e:
                print(f"The error '{e}' occurred when connecting for reading")
                print("Trying to reconnect to the database in 5 seconds")

        if not write_con:
            try:
                write_con = psycopg2.connect(database=db_conf['NAME'],
                                            user=db_conf['USER'],
                                            password=db_conf['PASSWORD'],
                                            host=db_conf['HOST'])
                print("Write connection to PostgreSQL DB successful")
            except OperationalError as e:
                print(f"The error '{e}' occurred when connecting for writing")
                print("Trying to reconnect to the database in 5 seconds")

        time.sleep(5)

    limit = int(input('Введите размер выборки, напр. 1000, 5000, 10000: '))
    SELECT = f"SELECT id, fields, created_at from base LIMIT {limit}"  # переменная с запросом к базе данных
    # здесь мы непосредственно делаем запрос к бд и записываем результаты в переменную raw_contacts
    with read_con.cursor('select') as cur:
        cur.execute(SELECT)
        raw_contacts = {i: row for i, row in enumerate(cur)}

    read_con.close()

    print("Вот так выглядит один объект, полученный из базы: \n", raw_contacts[0])

    id_list = [{'id': raw_contacts[i]['id']} for i in range(len(raw_contacts))]  # получам словарь с id объектов

    for i in range(len(id_list)):  # создаём словарь со значения для обучения
        if raw_contacts[i]:
            parser = Parser()
            parser.get_values([raw_contacts[i]])
            id_list[i].update(parser.dict_of_values)
        else:
            parser = Parser()
            id_list[i].update(parser.dict_of_values)

    data_for_training = {i + 1: v for i, v in enumerate(id_list)}  # получаем итоговое представление данных для обучения

    print("Вот так выглядит объект, готовый для обучения модели: \n", data_for_training[1])

    # удаляем лишние словари и массивы
    del id_list, raw_contacts

    settings_file = 'test_settings_file'  # это файл с весами, или правилами, модели. Вам его читать не понадобится (и не получится)
    training_file = 'test_training_file.json'  # это файл с данными разметки. Его мы будем обновлять в процессе дообучения

    if os.path.exists(settings_file):  # переходим в эту ветку, если уже есть обученная модель и надо просто получить дубликаты
        answer = get_input(
            "У вас уже обученная модель. Вы хотите найти дубли и занести их в таблицу с её помощью? (y/n): ",
            ['y', 'n']
        )
        if answer.lower() == 'y':
            with open(settings_file, 'rb') as f:
                matcher = dedupe.StaticDedupe(f)
            confidence_for_clustering = get_input(
                "Введите уровень уверенности для определения дублей (прим: Это значение от 0 до 1. Чем ближе к 1, тем выше точность модели. Рекомендуем использовать 0.9 для записи дублей в бд. Вы можете экспериментировать: ",
                type_=float,
                min_=0.0,
                max_=1.0
            )

            # Выполнить дедупликацию
            duplicates = matcher.partition(data_for_training, confidence_for_clustering)
            save_high_confidence_clusters_to_db(duplicates, data_for_training)  # сохранение кластеров с уровнем уверенности выше 0.9 в базу данных
            print(f"Сохранили")
            del duplicates
        else:
            delete_or_not = get_input(
            "Вы хотите удалить файл настроек для модели, чтобы потом обучить ее заново? (y/n): ",
            ['y', 'n']
        )
            if delete_or_not.lower() == 'y':
                os.remove(settings_file)
            else:
                print("Окей, всего хорошего!")

    else:  # переходим в эту ветку, если хотим обучить с нуля или дообучить модель
        fields = []  # задаем для модели поля, на которых она будет обучаться. Можно выставить приоритеты для полей
        array_of_fields = get_input(
            "Введите через запятую поля, на которых хотите обучить модель. Например, «phone», «email»: ",
            ['phone', 'email', 'created_at'],
            split=True
        )
        array_of_weights = get_input(
            "Через запятую укажите приоритеты для введённых полей. Например, если особенно важно поле «phone» поставьте ему 2, а «email» — 1. Вводите значения в том же порядке, что в предыдущем шаге: ",
            type_=int,
            split=True
        )

        for field, weight in zip(array_of_fields, array_of_weights):
            fields.append({'field': field, 'type': 'String', 'has missing': True, 'weight': weight})

        deduper = dedupe.Dedupe(fields)  # создаем объект Dedupe, то есть модель, которую будем обучать

        if os.path.exists(training_file):  # если уже есть какой-то тренировочный файл, то загружаем его без первичного обучения
            print('У нас уже есть файл с размеченными экземплярами. Запустим поиск дублей с его помощью ', training_file)
            with open(training_file) as tf:
                deduper.prepare_training(data_for_training, tf)
        else:  # если ничего нет, то проводим первичное обучение
            print("Подождите немного, пока модель готовится к первичной разметке")
            deduper.prepare_training(data_for_training)  # подготавливаем модель к первичной разметке, нужно подождать. Чем
            # больше полей было указано для обучения, тем больше времени займет подготовка к обучению
            print(
                'Сейчас вам нужно будет сравнивать контакты и отвечать на вопрос: дубли это или нет. Введите y, если «да», n, если «нет», u, если вы не уверены. Когда закончите, введите f. PS: оценить стоит хотя бы несколько десятков пар')
            dedupe.console_label(deduper)

            # запишем результаты активного обучения в файл
            with open(training_file, 'w') as tf:
                deduper.write_training(tf)

        print("Модель готова к обучению")
        deduper.train()  # здесь мы собственно обучаем модель на размеченных данных. может потребоваться какое-то время
        print("Приступаем к кластеризации")

        clastering(deduper, data_for_training) # проводим кластеризацию и доразметку

        print("Работа завершена. Сохраняем настройки модели")

        with open(settings_file, 'wb') as sf: #записываем настройки модели в файл
            deduper.write_settings(sf)
