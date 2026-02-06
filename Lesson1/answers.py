import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
headers = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age', 'Diabetes']
dataset = pandas.read_csv(url, names=headers)
max_pregnancies = dataset['Pregnancies'].max()
mean_age = dataset['Age'].mean()
num_diabetes = dataset['Diabetes'].sum()
num_of_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)