from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
# 30.02, 59.19, 1021.00, 95.5
#0.03237, 0, 2.18, 0, 0.458, 6.998, 45.8, 6.0622, 3, 222, 18.7, 394.63, 2.94

class Regressors:
    def __init__(self, df):
        self.regressors = [LinearRegression(),  PolynomialFeatures(degree=4), tree.DecisionTreeRegressor(), RandomForestRegressor()]
        self.x = df.iloc[:, :-1]
        self.y = df.iloc[:, -1]
        self.models = {}
        self.predictions = {}
        self.best_model = {}

    def preprocces(self):
        imputer = KNNImputer(n_neighbors=7)
        for column in self.x:
            self.x[column] = imputer.fit_transform(self.x[[column]])
        self.y = imputer.fit_transform(np.array(self.y).reshape(-1, 1))
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.20)

    def train(self, x_data):
        print(x_data)
        self.preprocces()
        for model in self.regressors:
            if model.__class__.__name__ == 'PolynomialFeatures':
                lin_reg_with_poly = LinearRegression()
                lin_reg_with_poly.fit(model.fit_transform(self.x_train), self.y_train)
                # self.print_results(model=lin_reg_with_poly, poly=model)
                self.models[model.__class__.__name__] = lin_reg_with_poly.score(model.fit_transform(self.x_test), self.y_test)
                self.predictions[model.__class__.__name__] = lin_reg_with_poly.predict(model.fit_transform([x_data]))[0][0]
            else:
                model.fit(self.x_train, self.y_train)
                # self.print_results(model)
                self.models[model.__class__.__name__] = model.score(self.x_test, self.y_test)
                if model.__class__.__name__ == 'LinearRegression':
                    self.predictions[model.__class__.__name__] = model.predict([x_data])[0][0]
                else:
                    self.predictions[model.__class__.__name__] = model.predict([x_data])[0]
        self.ann(x_data)
        self.best_model['model'] = max(self.models)
        self.best_model['score'] = self.models[max(self.models)]
        self.best_model['prediction'] = round(self.predictions[max(self.models)], 2)
        return self.best_model

    def print_results(self, model, poly=False):
        if poly:
            print('Polynomial Features MODEL PERFORMANCE:')
            predicted_price = model.predict(poly.fit_transform(self.x_test))
            print(f'Polynomial Features Model R2 Score: {model.score(poly.fit_transform(self.x_test), self.y_test)}')
            print('Polynomial Features Model Mean Squared Error: ',
                  mean_squared_error(self.y_test, predicted_price))
            print()
        else:
            print(f'{model.__class__.__name__} MODEL PERFORMANCE:')
            predicted_price = model.predict(self.x_test)
            print(f'{model.__class__.__name__} Model R2 Score: {model.score(self.x_test, self.y_test)}')
            print(f'{model.__class__.__name__} Model Mean Squared Error: ', mean_squared_error(self.y_test, predicted_price))
            print()

    def ann(self, data):
        ann = tf.keras.models.Sequential()
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=1))
        ann.compile(optimizer='adam', loss='mean_squared_error', metrics=[self.r_squared])
        history = ann.fit(self.x_train, self.y_train, batch_size=32, epochs=100)
        self.models['ANN'] = history.history['r_squared'][-1]
        self.predictions['ANN'] = ann.predict([data]).tolist()[0][0]
        print('//////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA///////')
        print(self.predictions)

    def r_squared(self, y_true, y_pred):
        tss = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        rss = tf.reduce_sum(tf.square(y_true - y_pred))
        r_squared = 1 - (rss / tss)
        return r_squared
