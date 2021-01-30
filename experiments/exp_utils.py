from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report

def model_report(
    model,
    x_test,
    y_test,
    binary = True
):
    preds = model.predict(x_test)
    
    acc = accuracy_score(y_test, preds)
    conf_mat = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    print(f'Accuracy Score: {acc}')

    if binary:
        try:
            probs = model.predict_proba(x_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
            print(f'AUC: {auc}')
        except:
            pass

    print('Confusion Matrix:')
    print(conf_mat)

    print('Classification Report:')
    print(classification_report(y_test, preds))
