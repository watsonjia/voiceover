from EvalClassifier.BasicCNN import BasicCNN


if __name__ == '__main__':
    cnn = BasicCNN()
    cnn.train_straw()
    cnn.train_voiceover()
