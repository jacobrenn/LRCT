from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report


def model_report(
        model,
        x_test,
        y_test,
        binary=True,
        neural_net=False
):
    if neural_net:
        probs = model.predict(x_test)
        if binary:
            probs = probs.flatten()
            preds = (probs >= 0.5).astype(int)
        else:
            preds = probs.argmax(axis=1)

    else:
        preds = model.predict(x_test)
        try:
            probs = model.predict_proba(x_test)[:, 1]
        except:
            pass

    acc = accuracy_score(y_test, preds)
    conf_mat = confusion_matrix(y_test, preds)

    print(f'Accuracy Score: {acc}')

    if binary:
        try:
            auc = roc_auc_score(y_test, probs)
            print(f'AUC: {auc}')
        except:
            pass

    print('Confusion Matrix:')
    print(conf_mat)

    print('Classification Report:')
    print(classification_report(y_test, preds))
