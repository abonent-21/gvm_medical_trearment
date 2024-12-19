import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QFormLayout, QComboBox, QPushButton, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MentalHealthApp(QWidget):
    def __init__(self):
        super().__init__()

        # Загрузка данных и подготовка модели
        self.df = pd.read_csv("mental_health_diagnosis_treatment_.csv")
        self.df.drop(['Patient ID', 'Treatment Start Date', 'Treatment Duration', 'AI-Detected Emotional State', 'Therapy Type'], axis=1, inplace=True)

        # Преобразование категориальных переменных в числовые значения
        self.df['Outcome'] = self.df['Outcome'].astype('category').cat.codes
        self.df['Gender'] = self.df['Gender'].astype('category').cat.codes
        self.df['Diagnosis'] = self.df['Diagnosis'].astype('category').cat.codes
        self.df['Medication'] = self.df['Medication'].astype('category').cat.codes

        # Нормализация и дискретизация данных
        numerical_columns = ['Symptom Severity', 'Mood Score', 'Sleep Quality', 'Physical Activity', 'Stress Level', 'Treatment Progress']
        self.df['Age'] /= 100
        self.df['Adherence to Treatment'] /= 100
        for col in numerical_columns:
            self.df[col] /= 10
        for col in numerical_columns:
            self.df[col] = self.df[col].apply(lambda x: 0 if x < 0.33 else (1 if x < 0.67 else 2))
        self.df['Adherence to Treatment'] = self.df['Adherence to Treatment'].apply(lambda x: 0 if x < 0.33 else (1 if x < 0.67 else 2))
        self.df['Age'] = self.df['Age'].apply(lambda x: 0 if x < 0.18 else (1 if x < 0.35 else (2 if x < 0.60 else 3)))

        # Создание и обучение модели
        self.model = BayesianNetwork([
            ('Diagnosis', 'Medication'),
            ('Diagnosis', 'Symptom Severity'),
            ('Medication', 'Outcome'),
            ('Medication', 'Treatment Progress'),
            ('Adherence to Treatment', 'Treatment Progress'),
            ('Treatment Progress', 'Outcome'),
            ('Stress Level', 'Mood Score'),
            ('Sleep Quality', 'Mood Score'),
            ('Physical Activity', 'Mood Score'),
            ('Mood Score', 'Symptom Severity'),
            ('Symptom Severity', 'Outcome'),
            ('Age', 'Outcome'),
            ('Gender', 'Outcome')
        ])
        self.model.fit(self.df, estimator=MaximumLikelihoodEstimator)
        self.inference = VariableElimination(self.model)

        # Инициализация пользовательского интерфейса
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Mental Health Diagnosis and Medication Prediction')

        # Основной layout
        layout = QVBoxLayout()

        # Форма для ввода данных
        form_layout = QFormLayout()

        # Поля для ввода значений
        self.gender_input = QComboBox()
        self.gender_input.addItems(['', 'Male', 'Female'])
        self.sleep_quality_input = QComboBox()
        self.sleep_quality_input.addItems(['', 'Low', 'Medium', 'High'])
        self.stress_level_input = QComboBox()
        self.stress_level_input.addItems(['','Low', 'Medium', 'High'])
        self.age_input = QComboBox()
        self.age_input.addItems(['', '0-18', '18-35', '35-60', '60+'])
        self.diagnosis_input = QComboBox()
        self.diagnosis_input.addItems(['Bipolar Disorder', 'Generalized Anxiety', 'Major Depressive Disorder', 'Panic Disorder'])
        self.physical_activity = QComboBox()
        self.physical_activity.addItems(['', 'Low', 'Medium', 'High'])

        # Добавление элементов в форму
        form_layout.addRow('Gender:', self.gender_input)
        form_layout.addRow('Sleep Quality:', self.sleep_quality_input)
        form_layout.addRow('Stress Level:', self.stress_level_input)
        form_layout.addRow('Age:', self.age_input)
        form_layout.addRow('Diagnosis:', self.diagnosis_input)
        form_layout.addRow('Physical Activity', self.physical_activity)

        # Кнопка для получения предсказания
        self.result_label = QLabel('Prediction for Medication will appear here.')
        query_button = QPushButton('Get Medication Prediction')
        query_button.clicked.connect(self.on_query)

        # Добавление формы и кнопки в layout
        layout.addLayout(form_layout)
        layout.addWidget(query_button)
        layout.addWidget(self.result_label)

        # Добавление области для диаграммы
        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.show()

    def on_query(self):
        # Извлечение значений из полей ввода
        gender = self.gender_input.currentIndex()  # 0: Male, 1: Female
        sleep_quality = self.sleep_quality_input.currentIndex()  # 0: Low, 1: Medium, 2: High
        stress_level = self.stress_level_input.currentIndex()  # 0: Low, 1: Medium, 2: High
        age = self.age_input.currentIndex()  # 0: 0-18, 1: 18-35, 2: 35-60, 3: 60+
        diagnosis = self.diagnosis_input.currentIndex()  # 0, 1, 2, ...
        physical_activity = self.physical_activity.currentIndex()  # 0: Negative, 1: Positive

        # Формирование evidence для запроса
        entities = [(gender, 'Gender'), 
                    (sleep_quality, 'Sleep Quality'),
                    (stress_level, 'Stress Level'), 
                    (age, 'Age'),
                    (diagnosis, 'Diagnosis'), 
                    (physical_activity, 'Physical Activity')
                    ]
        
        evidence = {'Outcome': 1}

        for item in entities:
            if item[1] == 'Diagnosis':
                evidence['Diagnosis'] = item[0]
            elif item[0] != 0:
                evidence[item[1]] = item[0] - 1


        # Запрос к модели для предсказания Medication
        result = self.inference.query(variables=['Medication'], evidence=evidence)
        
        # Отображаем результат
        medication_dict = {
            0: 'Antidepressants',
            1: 'Antipsychotics',
            2: 'Anxiolytics',
            3: 'Benzodiazepines',
            4: 'Mood Stabilizers',
            5: 'SSRIs'
        }
        predicted_medication = medication_dict.get(result.values.argmax(), 'Unknown')
        self.result_label.setText(f"Predicted Medication: {predicted_medication}")
        
        # Диаграмма с вероятностями препаратов
        self.figure.clf()  # Полностью очищаем фигуру
        ax = self.figure.add_subplot(111)
        ax.clear()
        
        # Подготовка данных для диаграммы
        probabilities = result.values
        medications = [medication_dict[i] for i in range(len(probabilities))]
        
        # Построение столбчатой диаграммы
        ax.bar(medications, probabilities, color='skyblue')
        ax.set_title("Probability Distribution of Medications")
        ax.set_xlabel("Medication")
        ax.set_ylabel("Probability")
        
        # Обновление диаграммы
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MentalHealthApp()
    sys.exit(app.exec_())
