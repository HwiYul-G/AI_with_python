from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np

digit = datasets.load_digits()
s = svm.SVC(gamma=0.001)
# 첫 번재 매개변수는 분류기 모델에 해당한는 객체,
# 두 번째, 세 번째 매겨변수는 특징 벡터와 레이블 정보
# 네 번째 매개변수는 5이므로 5겹 겹차검증을 수행
accuracies = cross_val_score(s, digit.data, digit.target, cv=5) # 5겹 교차검증

print(accuracies)
print("정확률(평균)= %0.3f, 표준편차=%0.3f" %(accuracies.mean()*100, accuracies.std()))