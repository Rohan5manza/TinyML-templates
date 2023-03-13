
from sklearn import svm
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

faces=fetch_lfw_people(min_faces_per_person=60,resize=0.4)

X_train,X_test,y_train,y_test=train_test_split(faces.data,faces.target,test_size=0.25,random_state=42)


clf=svm.SVC(kernel='linear',C=1).fit(X_train,y_train)

score=clf.score(X_test,y_test)

print("Accuracy: ",score)

