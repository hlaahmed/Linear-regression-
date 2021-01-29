import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

#Use Pandas library to read the dataset text file
data_set = pd.read_csv("train_data_file.txt", header=None, delim_whitespace=True)
X = data_set[0]
y = data_set[1]

def model():
    # An instance of the model is created.
    # keras.Sequential has one input and one output, so is appropriate to use in Linear Regression
    model = tf.keras.Sequential()

    # Setting the the number of outputs = 1 which is "y"
    # Input Shape = 1 which is X
    model.add(tf.keras.layers.Dense(1, input_shape=[1]))

    # The model calculates the Mean Squared Error which is the loss function of the Linear Regression
    # The model uses the stochastic gradient with a learning rate = 0.01
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

    # Training the data_set with 300 epochs (iterations)
    training = model.fit(X, y, epochs=300)

    # After training the data, we now test our model,
    # By passing the X values as inputs and letting the predict function predict the output.
    data_set["Predicted Output"] = model.predict(X)

model()
#Plot the Trained Dataset and the predicted output.
plt.scatter(X, y)
plt.plot(X, data_set["Predicted Output"], color='r')
plt.show()





