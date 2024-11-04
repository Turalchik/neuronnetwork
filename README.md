# Проект Makural

Проект **Makural** представляет собой реализацию многослойной нейронной сети на C++ с возможностью обучения на данных MNIST и сохранения модели. Сеть поддерживает выбор различных функций активации и функций ошибки, а также оптимизаторы `SGD` и `Adam`.

## Основные файлы и классы

- `makural.h` - содержит основной класс `NeuralNetwork` с методами обучения, предсказания и сохранения модели.
- `layer.h` - объявление классов для различных слоёв сети.
- `costfunctions.h` - содержит функции для вычисления стоимости.
- `mnistools.h` - методы загрузки и предобработки данных из MNIST.
  
## Основные функции и методы

### NeuralNetwork
Класс `NeuralNetwork` предоставляет API для создания, обучения и использования нейронной сети:

- **Конструкторы**
  - `NeuralNetwork(const std::vector<int>& layers_sizes, const char* general_activation_function, const char* output_activation_function, const char* cost_function)`
    Создаёт сеть с заданными слоями и функциями активации.
  - `NeuralNetwork(const char* input_filename)` - загружает сеть из файла.

- **Методы**
  - `void train(dataVec& data_train, dataVec& answers_train, const double& validation_split, const char* optimizer_name, const size_t& epochs, const size_t& butchSize)` - обучение сети.
  - `Eigen::MatrixXd answerVec(const Eigen::MatrixXd& input)` - получение предсказания.
  - `int predict(const Eigen::MatrixXd& input)` - предсказание класса.
  - `void save(const std::string& output_filename) const` - сохранение модели.
  - `double accuracy(const dataVec& data_test, const dataVec& answers_test)` - вычисление точности на тестовых данных.
  - `double averageLoss(const dataVec& data_test, const dataVec& answers_test, const size_t& from, const size_t& to)` - среднее значение ошибки на выборке.

### Методы оптимизации

- **Stochastic Gradient Descent (SGD)** - реализован в методе `optimizerSGD`.
- **Adam** - метод адаптивного обучения, реализованный в `Adam`.

### Загрузка и предобработка данных

Для загрузки и предобработки изображений из MNIST, а также конвертации PNG изображений в матрицы используется метод `MNISTLoader` и функция `pngToMatrix`.

## Пример использования

### Инициализация и обучение

```cpp
#include "makural.h"

int main() {
    // Определение размеров слоёв
    std::vector<int> layers = {784, 128, 64, 10};
    
    // Создание экземпляра сети
    NeuralNetwork nn(layers, "relu", "softmax", "cross_entropy");

    // Загрузка данных
    std::ifstream dataFile("mnist_train.csv");
    std::vector<Eigen::MatrixXd*> data_train;
    std::vector<Eigen::MatrixXd*> answers_train;
    MNISTLoader(dataFile, data_train, answers_train);
    
    // Обучение сети
    nn.train(data_train, answers_train, 0.2, "sgd", 10, 32);
    
    // Сохранение модели
    nn.save("model.bin");
    
    return 0;
}
```

## Прогноз на новом изображении

```cpp
#include"makural.h"
#include "mnistools.h"

int main() {
	NeuralNetwork makural("MNIST.txt");
	Eigen::MatrixXd res = pngToMatrix("pic.png");
	std::cout << makural.predict(res) << " " << makural.answerVec(res).maxCoeff() * 100 << "%";

	return 0;
}
```

## Требования

Для компиляции и запуска проекта потребуются:

Eigen - библиотека для работы с матрицами.
OpenCV - библиотека для предобработки изображений.


## Компиляция

Проект можно скомпилировать с использованием g++ или другого компилятора C++ с поддержкой стандартов C++11 или выше.

```bash
g++ -o neural_net main.cpp -I/path_to_eigen -I/path_to_opencv -lopencv_core -lopencv_imgproc -lopencv_highgui
```

## Примечания

* В проекте предусмотрены различные функции активации и ошибки, которые можно задавать при инициализации сети. Например, для скрытых слоёв можно использовать функции активации типа ReLU или Sigmoid, а для выходного слоя - Softmax или Linear.

* Обучение можно настраивать, задавая оптимизатор и размер пакета для стохастического градиентного спуска или Adam. При этом также поддерживаются параметры настройки скорости обучения (learning_rate) и эпохи (epochs), что позволяет гибко адаптировать обучение под конкретные задачи и объём данных.

* Для удобства загрузки данных в проекте предусмотрен загрузчик MNISTLoader, который может загружать датасеты формата MNIST и преобразовывать изображения в пригодный для использования вид.

* Методы сохранения и загрузки позволяют сохранять обученную модель в файл и использовать её без повторного обучения.