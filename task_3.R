rm(list=ls())

# Установка необходимых библиотек
install.packages("mlbench")
install.packages("caret")
install.packages("kknn")
install.packages("vegan")

# Загрузка необходимых библиотек
library(mlbench)
library(caret)
library(kknn)
library(vegan)

# Загрузка датасета
data("BreastCancer")

for (i in 1:(ncol(BreastCancer)-1)) {
  # Преобразование столбцов к числовому типу
  BreastCancer[, i] <- as.numeric(BreastCancer[, i])
  # Избавляемся от NA-значений методом заменой средним значением
  BreastCancer[is.na(BreastCancer[, i]), i] <- mean(BreastCancer[,i], na.rm=TRUE)
}


View(BreastCancer)

head(BreastCancer$Class)

# Выбор индексов для обучающей выборки
# y - столбец, по которому классифицируются данные
# p - вероятность попадания индекса i-ой строки в trainIndex
# list - НЕ возвращаем список или вектор
trainIndex <- createDataPartition(y=BreastCancer$Class, p=0.8, list=FALSE)

trainSet <- BreastCancer[trainIndex, ] # обучающие данные
testSet <- BreastCancer[-trainIndex, ] # тестовые данные

# Настраиваем параметры обучения
# repeatedcv - повторяющийся метод кросс-валидации
# repeats - количество повторений
control <- trainControl(method = "repeatedcv", repeats = 3)

# Вычисление и выборка наилучших комбинации гиперпараметров
# Для целевого метода kknn
# Переменная Class зависит от остальных переменных модели
# tuneLength - количество комбинаций гиперпараметров модели
# ВЫПОЛНЯЕТСЯ ОЧЕНЬ ДОЛГО, ЗАВИСИТ ОТ tuneLength.
result <- train(Class~., data=trainSet, method = "kknn", 
                trControl=control, tuneLength=20)

summary(result) # Смотрим на Best k: 17

# Устанавливаем из summary
best_k <- 17

# Обучение модели методом kknn
# kernel - тип ядра для взвешивания соседей
model <- kknn(Class~., train=trainSet, test=testSet, k=best_k, kernel="rectangular")


# Построение таблицы сопряженности
confusion_matrix <- confusionMatrix(table(predict(model), testSet$Class)

# Находим точность модели
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

print(paste("Точность:", round(accuracy * 100, 2), "%"))

# Наоходим долю ошибки
error_rate <- 1-accuracy

print(paste("Величина ошибки:", round(error_rate * 100, 2), "%"))

# Выводы
# Лучшее значение k = 17
# По таблице сопряженности видно, что для класса benign модель сделала
# 89 правильных предсказания и 2 неверных, а для malignant - 46 правильных
# и 2 неправильных
# Процент ошибки составил всего 2.88 %. Точность модели - 97.22 %
# Модель вполне точно классифицирует данные. 




