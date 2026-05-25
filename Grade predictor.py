import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Features:
# [study_hours, attendance, practice_tests]

X = np.array([
    [1.0, 75, 3],
    [1.5, 80, 5],
    [2.0, 82, 6],
    [2.5, 85, 8],
    [3.0, 88, 10],
    [3.5, 90, 12],
    [4.0, 92, 15],
    [4.5, 94, 18],
    [5.0, 96, 20],
    [5.5, 98, 24],

    [0.5, 70, 2],
    [1.2, 78, 4],
    [1.8, 81, 5],
    [2.2, 84, 7],
    [2.8, 86, 9],
    [3.2, 89, 11],
    [3.8, 91, 14],
    [4.2, 93, 17],
    [4.8, 95, 21],
    [5.3, 97, 25],

    [1.0, 90, 4],
    [2.0, 91, 7],
    [3.0, 93, 13],
    [4.0, 95, 16],
    [5.0, 97, 23]
])
# Target:
# 1 = likely A*
# 0 = unlikely A*

y = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,

    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,

    0, 0, 1, 1, 1
])
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, predictions))

# Predict for a new student
studyhours= int(input("Enter the number of hours you study per day:"))
attendance= int(input("Enter your attendance(in %" "for whole year):"))
practicetest= int(input("Enter the number of practice tests attempted:"))
new_student = [[studyhours, attendance, practicetest]]

result = model.predict(new_student)
probability = model.predict_proba(new_student)

print("\nProbability of A*:", probability[0][1])

if result[0] == 1:
    print("Likely to get A*")
else:
    print("Unlikely to get A*")