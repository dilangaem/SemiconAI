import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score
import pickle, os
from datetime import datetime

seed=7
np.random.seed(seed)

class RFCModel:

    # init method or constructor
    def __init__(self, file_name='data.csv',crystal_sys='all',test_size=0.02):
        #Name of the data file
        self.file_name = file_name
        #'all' for all the crystal systems
        #Other seven arguments are as follows
        #'monoclinic', 'triclinic', 'orthorhombic', 'trigonal',
        #'hexagonal', 'cubic', 'tetragonal'
        self.crystal_sys=crystal_sys
        #test size of the test set
        self.test_size=test_size

    def load_data(self):
        df = pd.read_csv(self.file_name, header=None)
        #df = df.drop([9])
        df = df.dropna()

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(df.values.tolist())
        data_list = imp.transform(df.values.tolist())
        df = pd.DataFrame(data_list)

        if(self.crystal_sys=='monoclinic'):
            # Selecting only the monoclinic materials
            df = df.loc[df[1] == 1]
        elif (self.crystal_sys == 'triclinic'):
            # Selecting only the triclinic materials
            df = df.loc[df[2] == 1]
        elif (self.crystal_sys == 'orthorhombic'):
            # Selecting only the orthorhombic materials
            df = df.loc[df[3] == 1]
        elif (self.crystal_sys == 'trigonal'):
            # Selecting only the trigonal materials
            df = df.loc[df[4] == 1]
        elif (self.crystal_sys == 'hexagonal'):
            # Selecting only the hexagonal materials
            df = df.loc[df[5] == 1]
        elif(self.crystal_sys=='cubic'):
            # Selecting only the cubic materials
            df = df.loc[df[6] == 1]
        elif (self.crystal_sys == 'tetragonal'):
            # Selecting only the tetragonal materials
            df = df.loc[df[7] == 1]
        elif (self.crystal_sys == 'all'):
            # Selecting  the all materials
            pass

        return df

    def split_data(self,df):
        data_X, data_y = df.iloc[:, 1:].values, df.iloc[:, 0].values

        train_x, test_x, train_y, test_y = train_test_split(data_X, data_y, test_size=self.test_size, random_state=42)

        return (train_x, test_x, train_y, test_y)

    def normalize_data(self,data):
        train_x, test_x, train_y, test_y =data

        print(train_x.shape)
        print(train_y.shape)

        print(test_x.shape)
        print(test_y.shape)

        xx = abs(train_x)
        maxm = xx.max(axis=0)
        maxm[maxm == 0.0] = 1

        train_x /= maxm
        test_x /= maxm

        return (train_x, test_x, train_y, test_y)

    def run_ml(self,data):
        train_x, test_x, train_y, test_y = data

        scores = []
        clf = RandomForestClassifier(n_estimators=500, min_samples_split=10, min_samples_leaf=3, max_features='auto',
                                     max_depth=70, bootstrap=False)
        scores = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')  # neg_mean_squared_error
        model = clf.fit(train_x, train_y)
        y_rbf_test = model.predict(test_x)

        y_rbf_train = model.predict(train_x)

        print(scores)
        print("Mean Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std()))

        return (train_y, y_rbf_train, test_y, y_rbf_test, model)

    def print_clf_report(self,results):
        train_y, y_rbf_train, test_y, y_rbf_test, model =results

        print(classification_report(test_y, y_rbf_test))

    def save(self):
        dir='Saved_Model'
        now=datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{dir}/model-{now}.sav'
        if not os.path.exists(dir):
            os.makedirs(dir)

        # save the model to disk
        pickle.dump(self, open(filename, 'wb'))

if __name__ == '__main__':
    rfc_model=RFCModel(crystal_sys='cubic')
    df=rfc_model.load_data()
    data = rfc_model.split_data(df)
    norm_data=rfc_model.normalize_data(data)
    results=rfc_model.run_ml(norm_data)
    rfc_model.print_clf_report(results)
    rfc_model.save()
