### Dedupe: алгоритм поиска дублей в базе данных контактов, выгруженных с amoCRM.

#### Общее описание

Библиотека Dedupe [определяет](https://docs.dedupe.io/en/latest/how-it-works/Matching-records.html), принадлежат ли разные записи одному и тому же объекту. Для этого алгоритм сравнивает строки. Разность между ними вычисляется при помощи расстояния Хэмминга, точнее одной из разновидностей метода. Похожие строки алгоритм объединяет в кластеры, для их классификации используется регуляризованная логистическая регрессия.

Соответственно, программа выполняет несколько шагов. Первым делом она получает набор записей контактов из amoCRM. Предполагается, что они хранятся в виде таблицы в базе данных. Затем программа предобрабатывает записи, чтобы получить массив словарей, в которых значениями выступают некоторые характеристики контакта, например, номер телефона и email. По этому массиву Dedupe проводит кластеризацию. Если это первое взаимодействие пользователя с программой, ему рекомендуется провести первичную разметку полученных кластеров. С помощью встроенных средств Dedupe пользователю предлагается отметить, принадлежат ли объединённые алгоритмом в группы контакты одному объекту. Данные разметки сохраняются в json-файл (test_training_file.json).

После этого программа проводит кластеризацию с учётом разметки. Применяется иерархическая кластеризация с центроидами. Для кластеризации необходимо указать порог, при котором алгоритм будет разбивать записи на группы. Это число в интервале от 0 до 1 (включая их). Для каждого кластера в dedupe вычисляется метрика «похожести» записей, находящихся в группе. Если мы выбираем порог, близкий к нулю, алгоритм будет выдавать в качестве итоговых кластеров слабо похожие друг на друга записи. Соответственно при пороге, близком к нулю, программа будет объединять только сильно похожие записи. Порог [следует выбирать](https://docs.dedupe.io/en/latest/how-it-works/Choosing-a-good-threshold.html), исходя из ответа на вопрос, что важнее: охватить все возможные дубликаты, даже потеряв при этом точность, или выделить только те дубликаты, в которых модель наиболее «уверена», потеряв при этом полноту.

После кластеризации программа записывает дубли по порогу 0.9 в базу данных (выбранный порог можно менять по той же логике, что была описана выше). В таблице появляется колонка cluster с номером кластера, которому принадлежит определённый контакт. Группировка по значениям этой колонки позволяет просмотреть дубликаты. После записи «уверенных» кластеров в таблицу, программа предлагает дообучить модель, разметив те кластеры, метрика у которых оказалась ниже «уверенного» порога. Если пользователь выбирает этот вариант, он снова видит интерфейс разметки. Новыми данными о разметке обновляется тренировочный json-файл. В конце программа записывает параметры модели в файл settings. Пока он есть, модель можно повторно использовать на новых записях без обучения.

#### Сценарии программы.

Если есть файл settings, программа первым делом предложит использовать его для кластеризации записей. Можно воспользоваться этим, также можно удалить файл settings и обучить модель заново.

Если файла settings нет, запустится процесс обучения модели. Здесь есть два варианта: имеется тренировочный json-файл или не имеется. В первом случае программа предложит провести кластеризацию с его помощью или удалить файл. При выборе кластеризации можно будет в дальнейшем дообучить модель. Во втором случае запустится процесс первичной разметки. 

#### Работа с программой.

Все необходимые библиотеки перечислены в requirements.txt. Программа написана на python. В качестве базы данных используется PostgreSQL. Необходимо настроить подключение к ней для корректной работы (нужны имя БД, пароль и т.д.). Для работы программы запускается файл main.py. Интерфейс доступен в консоли. 

#### Как можно улучшить?

При работе с большим количеством записей (контактов) Dedupe может обрабатывать данные медленно. Процесс подготовки к обучению и процесс кластеризации могут занимать существенное время. Необходимость оптимизации нужно рассматривать в зависимости от задач масштабирования и аппаратных характеристик сервера. Для больших датасетов Dedupe предлагает считывать с базы данных PostgresSQL пары для сравнения по частям. Подробнее почитать об этом и посмотреть пример можно по [этой ссылке](https://dedupeio.github.io/dedupe-examples/docs/pgsql_big_dedupe_example.html).

#### Ссылки.

- [Документация Dedupe](https://docs.dedupe.io/en/latest/API-documentation.html)
- [Документация по работе с контактами amoCRM](https://www.amocrm.ru/developers/content/crm_platform/contacts-api)

