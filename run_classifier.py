from EvalClassifier.BasicCNN import BasicCNN


def show_roc(straw_model, straw_x, straw_y, voice_model, voice_x, voice_y):
    from sklearn.metrics import roc_curve, auc
    y_pred = straw_model.predict(straw_x)
    fpr_straw, tpr_straw, thresholds = roc_curve(straw_y, y_pred)
    auc_straw = auc(fpr_straw, tpr_straw)

    y_pred = voice_model.predict(voice_x)
    fpr_voice, tpr_voice, thresholds = roc_curve(voice_y, y_pred)
    auc_voice = auc(fpr_voice, tpr_voice)

    from matplotlib import pyplot as plt
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_straw, tpr_straw, label='Strawman (auROC = {:.3f})'.format(auc_straw))
    plt.plot(fpr_voice, tpr_voice, label='Voiceover (auROC = {:.3f})'.format(auc_voice))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    # Zoom in view of the upper left corner.
    # plt.figure(2)
    # plt.xlim(0, 0.2)
    # plt.ylim(0.8, 1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve (zoomed in at top left)')
    # plt.legend(loc='best')
    # plt.show()


if __name__ == '__main__':
    cnn = BasicCNN()
    straw_results = cnn.train_straw()
    voice_results = cnn.train_voiceover()

    show_roc(
        straw_results[0], straw_results[1], straw_results[2],
        voice_results[0], voice_results[1], voice_results[2]
    )
