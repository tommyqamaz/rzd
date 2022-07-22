# Всероссийский чемпионат. Цифровой прорыв 2022.
Разработка алгоритма определения железнодорожной колеи и подвижного состава для предотвращения чрезвычайных ситуаций на железной дороге
## Решение
Так как я зарегистрировался в последний день, нужно было придумать простое (меньше кода) и быстрое решение (быстрое обучения на бесплатном инстансе kaggle). Тренировал модели на pytorch с использованием pytorch-lightning. Метрики отслеживал в [Weight and Biases](https://wandb.ai/site).
### Выбор модели
  Трансформеры показывает впечатляющие результаты (например [этот](https://arxiv.org/abs/2105.15203)), но казалось бы, рельсы и поезда достаточно простые обьектые для сегментации (строгая однообразная геометрическая форма и низкая вариативность) и трансформеры будут слишком мощными и тяжелыми для такой задачи, поэтому в качестве бейзлайна принято выбрать что-то попроще, а к ним вернуться позже.

Изначально расматривал для решения [U-Net++](https://arxiv.org/pdf/1807.10165.pdf), но на картинках, ужатых до 512 на 512 с размером батча 8 эпоха занимала почти 3 часа (на nvidia p100 16 gb на Kaggle).
Потратив несколько часов на поиск и тест оптимальной модели, решил остановиться на классичeской U-Net.
[Предобученные модели брал отсюда.](https://github.com/qubvel/segmentation_models.pytorch#models).
### Выбор аугментаций.
Бейзлайн включал в себя U-Net c backbone [Efficientnet-b6](https://arxiv.org/abs/1905.11946)(с весами imagenet), Adam optimizer (lr=3e-4), без аугментаций и learning rate scheduler. В таком конфиге модель выдавала 0.68 mIoU за 3 эпохи.
Следующим этапом стал поиск подходящих аугментаций: к классическим HorizontalFLip, RandomBrightness and Contrast я добавил аугментации [для автомобильной сегментации](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library): случайные тени, случайные солнечные блики и туман, на этих аугментациях
модель выдавала 0.74 за те же 3 эпохи. Нормализация среднего и дисперсии картинок.
### Выбор loss-функции.
Модель тренировалась с Dice-loss, и последние две эпохи с Lovasz-loss, с ним удалось достигнуть 0.78 mIoU.

### Результат
Попал в топ-30 из около 200 участников.

## Что не успел реализовать. 😔
1. Patch segmentation: делать случайный кроп картинки и предсказывать на ней, затем с нахлестом проходиться по тестовым картинкам (плюсы: нет потери информации, минусы: долго).
2. Learning rate scheduler: [StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html), [CosineAnnealingLR](https://arxiv.org/pdf/1608.03983.pdf), CosineAnnealingWarmRestarts.
3. Большее количество эпох, так как даже на 5 эпохах лосс на валидационной выборке продолжает падать.
4. Confusion matrix для поиска ошибок и подсчёта метрик.
5. Ансамбль модей.
6. Segmentation Transformers.
7. Ужимать до большего чем 512x512 размера, так как при сжатии с 4k картинок теряется много информации.
8. Попробывать ставить разный learnig rate, я тренировал с 3e-4.
9. Модель сильно ошибается на ночных картинках, дотренировать модель на ночных фото или отказаться от нормализации.
10. Найти более мощную ВМ для быстрой проверки гипотез.
## Формальное описание хакатона. 🤓

### Разработка алгоритма определения железнодорожной колеи и подвижного состава для предотвращения чрезвычайных ситуаций на железной дороге
Автоматизация функций управления и обеспечения безопасности за счет внедрения технических средств — всё это способы повышения уровня безопасности на железнодорожном транспорте.

Внедрение новой техники и технологий автоматизации позволяет исключить некоторые опасные технологические операции и значительно изменить характер работы многих работников железной дороги. Внедрение блоков определения препятствий на основе видеоаналитики позволяет вести дополнительный визуальный контроль пространства перед поездом, определять путь и направление движения. Применение камер различного диапазона видимости и фокусного расстояния увеличивает возможности системы технического зрения и превышает возможности человека по скорости реакции, дальности определения препятствий в том числе в сложных погодных условиях (ночь, дождь, снег, туман, задымленность, ослепление солнечным светом, переход из затемненных участков на освещенные).

Создание интеллектуальных систем, предупреждающих машиниста о возможном столкновении с потенциально опасными объектами, содержит в себе несколько первостепенных задач: определения колеи и подвижного состава. В рамках чемпионата требуется создать алгоритм, определяющий элементы дорожной инфраструктуры: колею (рельсошпальную решетку) и подвижной состав (локомотивы, грузовые вагоны, пассажирские вагоны).
