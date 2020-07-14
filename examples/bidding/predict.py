from ludwig.api import LudwigModel


model_path = './results/simple_experiment_simple_model/model'
model = LudwigModel.load(model_path)

predictions = model.predict(data_csv='./data/test_0605.csv')
predictions.to_csv("data/answer.csv")
print(predictions)
