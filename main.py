import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Загрузка данных из Excel файла
data = {
    'Регион': [
        'Республика Узбекистан', 'Республика Каракалпакстан', 'Андижанская',
        'Бухарская', 'Джизакская', 'Кашкадарьинская', 'Навоийская',
        'Наманганская', 'Самаркандская', 'Сурхандарьинская',
        'Сырьдарьинская', 'Ташкентская', 'Ферганская', 'Хорезмская', 'г. Ташкент'
    ],
    '2013': [13.4, 15.8, 15.0, 12.8, 9.6, 11.6, 10.5, 14.1, 11.0, 14.2, 
             13.6, 14.0, 13.7, 12.7, 17.2]
}

# Создаем DataFrame
df = pd.DataFrame(data)

# Преобразуем данные для анализа
X = np.zeros((len(df), 1))  # Массив для данных 2013 года
names = []  # Список для названий регионов

# Заполняем массивы данными
for i, row in df.iterrows():
    names.append(row['Регион'])
    X[i, 0] = row['2013']

print("Данные по регионам Узбекистана (смертность детей до 5 лет на 1000 живорождений, 2013 год):")
print(df)
print("\nМассив данных для анализа:")
print(X)

# Визуализация исходных данных
plt.figure(figsize=(12, 6))

# Гистограмма смертности по регионам
plt.subplot(121)
bars = plt.barh(range(len(df)), X.flatten())
plt.yticks(range(len(df)), [name[:15] + '...' if len(name) > 15 else name for name in names])
plt.xlabel('Смертность детей до 5 лет (на 1000 родившихся живыми)')
plt.title('Смертность детей до 5 лет по регионам Узбекистана (2013)')
plt.grid(True, alpha=0.3)

# Добавляем значения на столбцы
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
             f'{X[i, 0]:.1f}', ha='left', va='center')

# Анализ данных
print("\n" + "="*60)
print("СТАТИСТИЧЕСКИЙ АНАЛИЗ ДАННЫХ:")
print("="*60)
print(f"Средняя смертность: {np.mean(X):.2f}")
print(f"Медианная смертность: {np.median(X):.2f}")
print(f"Минимальная смертность: {np.min(X):.2f} ({names[np.argmin(X)]})")
print(f"Максимальная смертность: {np.max(X):.2f} ({names[np.argmax(X)]})")
print(f"Стандартное отклонение: {np.std(X):.2f}")

# Кластеризация регионов по уровню смертности
print("\n" + "="*60)
print("КЛАСТЕРИЗАЦИЯ РЕГИОНОВ ПО УРОВНЮ СМЕРТНОСТИ:")
print("="*60)

# Так как у нас один признак (смертность в 2013), создадим второй "признак" - ранг региона
# Это позволит провести более осмысленную кластеризацию
X_cluster = np.zeros((len(df), 2))
for i in range(len(df)):
    X_cluster[i, 0] = X[i, 0]  # Смертность
    X_cluster[i, 1] = i  # Порядковый номер региона

# Выполняем кластеризацию K-means
n_clusters = 3
random_state = 42
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
y_pred = kmeans.fit_predict(X_cluster)

# Анализ кластеров
clusters = {}
for i, cluster in enumerate(y_pred):
    if cluster not in clusters:
        clusters[cluster] = []
    clusters[cluster].append((names[i], X[i, 0]))

print(f"\nРегионы разделены на {n_clusters} кластера:")
for cluster_id in sorted(clusters.keys()):
    cluster_regions = clusters[cluster_id]
    mortalities = [x[1] for x in cluster_regions]
    print(f"\nКластер {cluster_id + 1}:")
    print(f"  Количество регионов: {len(cluster_regions)}")
    print(f"  Средняя смертность: {np.mean(mortalities):.2f}")
    print(f"  Регионы:")
    for region, mortality in cluster_regions:
        print(f"    - {region}: {mortality:.1f}")

# Визуализация кластеризации
plt.subplot(122)
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Рисуем точки для каждого региона
for i in range(len(df)):
    plt.scatter(X_cluster[i, 0], i, 
                c=colors[y_pred[i] % len(colors)], 
                s=100, alpha=0.7, 
                label=f'Кластер {y_pred[i]+1}' if i == 0 else "")

# Наносим подписи
for i in range(len(df)):
    plt.annotate(names[i][:10] + ('...' if len(names[i]) > 10 else ''), 
                (X_cluster[i, 0], i), 
                xytext=(5, 0), textcoords='offset points')

plt.xlabel('Смертность детей до 5 лет (на 1000 родившихся живыми)')
plt.ylabel('Регион (индекс)')
plt.title('Кластеризация регионов по уровню смертности (2013)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Дополнительный анализ: тренды смертности
print("\n" + "="*60)
print("АНАЛИЗ РЕГИОНОВ С НАИБОЛЬШЕЙ И НАИМЕНЬШЕЙ СМЕРТНОСТЬЮ:")
print("="*60)

# Находим регионы с максимальной и минимальной смертностью
max_idx = np.argmax(X)
min_idx = np.argmin(X)

print(f"Наивысшая смертность: {names[max_idx]} - {X[max_idx, 0]:.1f}")
print(f"Наименьшая смертность: {names[min_idx]} - {X[min_idx, 0]:.1f}")
print(f"Разница: {X[max_idx, 0] - X[min_idx, 0]:.1f} ({((X[max_idx, 0] - X[min_idx, 0]) / X[min_idx, 0] * 100):.1f}%)")

# Классификация регионов по уровню смертности
print("\n" + "="*60)
print("КЛАССИФИКАЦИЯ РЕГИОНОВ ПО УРОВНЮ СМЕРТНОСТИ:")
print("="*60)

# Определяем пороговые значения
low_threshold = np.percentile(X, 33)  # Нижняя треть
high_threshold = np.percentile(X, 66)  # Верхняя треть

print(f"Пороговые значения:")
print(f"  Низкий уровень: < {low_threshold:.1f}")
print(f"  Средний уровень: {low_threshold:.1f} - {high_threshold:.1f}")
print(f"  Высокий уровень: > {high_threshold:.1f}")

low_regions = []
medium_regions = []
high_regions = []

for i in range(len(df)):
    mortality = X[i, 0]
    if mortality < low_threshold:
        low_regions.append((names[i], mortality))
    elif mortality > high_threshold:
        high_regions.append((names[i], mortality))
    else:
        medium_regions.append((names[i], mortality))

print(f"\nНизкий уровень смертности ({len(low_regions)} регионов):")
for region, mortality in low_regions:
    print(f"  - {region}: {mortality:.1f}")

print(f"\nСредний уровень смертности ({len(medium_regions)} регионов):")
for region, mortality in medium_regions:
    print(f"  - {region}: {mortality:.1f}")

print(f"\nВысокий уровень смертности ({len(high_regions)} регионов):")
for region, mortality in high_regions:
    print(f"  - {region}: {mortality:.1f}")

# Создаем итоговый отчет
print("\n" + "="*60)
print("ИТОГОВЫЙ ОТЧЕТ:")
print("="*60)
print(f"Год анализа: 2013")
print(f"Показатель: Смертность детей до 5 лет (на 1000 родившихся живыми)")
print(f"Количество регионов: {len(df)}")
print(f"Общая картина: {len(low_regions)} регионов с низкой смертностью, "
      f"{len(medium_regions)} со средней, {len(high_regions)} с высокой")
print(f"Критическая ситуация: {names[max_idx]} (в {X[max_idx, 0]/X[min_idx, 0]:.1f} раза выше, чем в {names[min_idx]})")
