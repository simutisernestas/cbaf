# from numpy import genfromtxt
# my_data = genfromtxt('eggs.csv', delimiter=',')
# my_data = np.flip(my_data, axis=0)
# my_data = my_data[:,1:]
# # my_data = np.expand_dims(my_data, -1)
# # my_data = np.swapaxes(my_data,1,2)
# # my_data.shape
# # X_train = my_data
# 100-my_data.shape[0]%100
# sq = 100
# missing = (my_data.shape[0])%sq
# my_data = my_data[:-missing]
# my_data = np.array(np.split(my_data, sq))
# my_data.shape
# my_data = np.swapaxes(my_data,0,1)
# # my_data = np.expand_dims(my_data, -1)
# my_data.shape
# X_train = my_data
# X_train.shape
# X_train[0][0]a = X_train
# # e = (a - np.mean(a)) / np.std(a)
# # could be done with a little bit more care :)
# x_normed = a / a.max(axis=0)
# # x_normed[1,:,4]
# # X_train[1,:,1]
# X_train = x_normed

# prev_cost = self.cost_basis * self.shares_held
# self.cost_basis = (prev_cost + paid_amount) / \
#     (self.shares_held + shares_bought)
