import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#Load Breast Cancer data from imported dataset
breast_cancer_data = load_breast_cancer()

#Split data into Training Data, Validation Data, Training Labels and Validation Labels, 80% of all data to Traing and 20% to Validate (Test)
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

#Finds the best K parameter and store it's value and it's scores
best_k = 0
best_score = 0
k_list = range(1, 100)
accuracies = []
for i in k_list:
  classifier = KNeighborsClassifier(n_neighbors = i)
  classifier.fit(training_data, training_labels)
  tmp = classifier.score(validation_data, validation_labels)
  accuracies.append(tmp)
  if(tmp > best_score):
    best_score = tmp
    best_k = i

#Set plotter to show our model results as K parameter changes
plt.title("Breast Cancer Classifier")
plt.xlabel("K-Value")
plt.ylabel("Accuracy")
plt.plot(k_list, accuracies)
plt.show()