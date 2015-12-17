
lstm = LSTMNetwork()

lstm.populate(1, [2, 1])

print("result: " + str(lstm.feedforward(np.array([1]))))
print("result: " + str(lstm.feedforward(np.array([1]))))
    #,
    #np.array([0]),
    #np.array([1])])
