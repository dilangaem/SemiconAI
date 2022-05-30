import RFClassifier
import FeatureEngine
import pandas as pd
import numpy as np
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    #getting the feature
    formula_file=f'DATA/{args.file_name}'
    feature_eng=FeatureEngine.Features(formula_file=formula_file)
    features=feature_eng.get_features()

    #getting the targets and symmetries
    df=pd.DataFrame(features)
    pred_x= df.iloc[:, 1:].values
    print(f'Number of materials for predicting: {len(df)}')

    rfc=RFClassifier.RFC()

    #provide the path to saved model in load_model
    loaded_model, maxm = rfc.load_model(f'TRAINED/{args.model_name}')

    #getting the chemical formulas
    df_mat=pd.read_csv(formula_file,header=None)
    formulas=df_mat.iloc[:,0].values
    rfc.predict(formulas=formulas,model=loaded_model,maxm=maxm,pred_x=pred_x)
    print('Please check the RESULTS directory for the predictions.')
