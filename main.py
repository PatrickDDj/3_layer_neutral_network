import BP_model as model
# 3 4 3
# 20 16 7
if __name__ == '__main__':
    print("鸢尾花数据----------------------------")
    Iris_X_train, Iris_Y_train, Iris_X_test, Iris_Y_test = model.load_Iris(p=0.8)
    Iris_parameters = model.fit(Iris_X_train, Iris_Y_train, n_x=4, n_h=3, n_y=3, iterations=20000, learning_rate=0.005)
    model.evaluate(Iris_parameters, Iris_X_test, Iris_Y_test)

    print("\n")

    print("动物园数据----------------------------")
    zoo_X_train, zoo_Y_train, zoo_X_test, zoo_Y_test = model.load_zoo(p=0.8)
    zoo_parameters = model.fit(zoo_X_train, zoo_Y_train, n_x=16, n_h=20, n_y=7, iterations=20000, learning_rate=0.005)
    model.evaluate(zoo_parameters, zoo_X_test, zoo_Y_test)