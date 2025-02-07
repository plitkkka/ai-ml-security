import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Загрузка и предобработка данных
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

# Создание модели
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
    return model

# Обучение модели
batch_size = 32
model = create_model()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=batch_size, validation_data=(test_images, test_labels))

# Оценка модели
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Точность на тестовых данных: {test_acc}')

# Предсказания на тестовых данных
test_predictions = model.predict(test_images)
test_predictions_classes = np.argmax(test_predictions, axis=1)

# Вычисление метрик F1-score, precision и recall
precision = precision_score(test_labels, test_predictions_classes, average='weighted')
recall = recall_score(test_labels, test_predictions_classes, average='weighted')
f1 = f1_score(test_labels, test_predictions_classes, average='weighted')

# Вывод метрик
print(
    f'Метрики модели на тестовых данных: Accuracy: {test_acc:.4f}, F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')


# Создание искажающего шаблона
def create_adversarial_pattern(model, images, labels):
    signed_grads = []

    for i in range(0, len(images), batch_size):
        image_batch = images[i:i + batch_size]
        label_batch = labels[i:i + batch_size]

        image_batch = tf.convert_to_tensor(image_batch)
        label_batch = tf.convert_to_tensor(label_batch)

        with tf.GradientTape() as tape:
            tape.watch(image_batch)
            prediction = model(image_batch)
            loss = tf.keras.losses.sparse_categorical_crossentropy(label_batch, prediction)

        gradient = tape.gradient(loss, image_batch)
        signed_grad = tf.sign(gradient)
        signed_grads.append(signed_grad)

    return tf.concat(signed_grads, axis=0)


# Генерация искажающих примеров для всего тестового набора
adversarial_test_images = test_images + create_adversarial_pattern(model, test_images, test_labels) * 0.1
adversarial_test_images = tf.clip_by_value(adversarial_test_images, 0, 1)

# Оценка на искаженных примерах
adversarial_predictions = model.predict(adversarial_test_images)
adversarial_predictions_classes = np.argmax(adversarial_predictions, axis=1)

# Вычисление точности на искажённых примерах
adversarial_accuracy = accuracy_score(test_labels, adversarial_predictions_classes)
print(f'Точность на искажённых данных: {adversarial_accuracy}')

# Вычисление метрик F1-score, precision и recall на искажённых данных
adversarial_precision = precision_score(test_labels, adversarial_predictions_classes, average='weighted')
adversarial_recall = recall_score(test_labels, adversarial_predictions_classes, average='weighted')
adversarial_f1 = f1_score(test_labels, adversarial_predictions_classes, average='weighted')

# Вывод метрик для искаженных данных
print(
    f'Метрики для искажённой модели: Accuracy: {adversarial_accuracy:.4f}, F1-score: {adversarial_f1:.4f}, Precision: {adversarial_precision:.4f}, Recall: {adversarial_recall:.4f}')

# Дистилляция модели
temperature = 10.0

# Обучение модели-учителя
soft_model = create_model()
soft_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
soft_model.fit(train_images, train_labels, epochs=5, batch_size=batch_size, validation_data=(test_images, test_labels))

# Получение "мягких" меток
soft_labels_train = tf.nn.softmax(soft_model.predict(train_images) / temperature)
soft_labels_test = tf.nn.softmax(soft_model.predict(test_images) / temperature)

# Генерация искажающих примеров для тренировочного набора
adversarial_train_images = train_images + create_adversarial_pattern(soft_model, train_images, train_labels) * 0.1
adversarial_train_images = tf.clip_by_value(adversarial_train_images, 0, 1)

# Обучение модели-дистиллятора
distilled_model = create_model()
distilled_model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
distilled_model.fit(adversarial_train_images, soft_labels_train, epochs=5, batch_size=batch_size,
                    validation_data=(adversarial_test_images, soft_labels_test))

# Оценка защищённой модели
test_loss, test_acc = distilled_model.evaluate(adversarial_test_images, soft_labels_test, verbose=2)
print(f'Точность дистиллированной модели: {test_acc}')

# Предсказания на тестовых данных защищённой модели
distilled_predictions = distilled_model.predict(adversarial_test_images)
distilled_predictions_classes = np.argmax(distilled_predictions, axis=1)

# Вычисление метрик F1-score, precision и recall для защищённого классификатора
distilled_precision = precision_score(test_labels, distilled_predictions_classes, average='weighted')
distilled_recall = recall_score(test_labels, distilled_predictions_classes, average='weighted')
distilled_f1 = f1_score(test_labels, distilled_predictions_classes, average='weighted')

# Вывод метрик для защищённого классификатора
print(
    f'Метрики для защищённого классификатора: Accuracy: {test_acc:.4f}, F1-score: {distilled_f1:.4f}, Precision: {distilled_precision:.4f}, Recall: {distilled_recall:.4f}')
