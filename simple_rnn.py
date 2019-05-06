import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class RNN(nn.Module):
	def __init__(self, input_size, output_size, hidden_dim, n_layers):
		super(RNN, self).__init__()
		self.hidden_dim = hidden_dim

		self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)


		self.fc = nn.Linear(hidden_dim, output_size)

	def forward(self, x, hidden):
		# shape of x is (batch_size, seq_length, input_size)
		# shape of hidden (n_layers, batch_size, hidden_dim)
		# r_out (batch_size, time_step, hidden_size)

		batch_size = x.size(0)

		r_out, hidden = self.rnn(x, hidden)

		r_out = r_out.view(-1, self.hidden_dim)

		output = self.fc(r_out)

		return output, hidden

def train(rnn, n_steps, print_every):
	#initialize hidden state
	hidden = None

	for batch_i, step in enumerate(range(n_steps)):
		# defining the training data
		time_steps = np.linspace(step * np.pi, (step+1)*np.pi, seq_length + 1)
		data = np.sin(time_steps)
		data.resize((seq_length + 1, 1)) # input_size=1

		x = data[:-1]
		y = data[1:]
		x_tensor = torch.Tensor(x).unsqueeze(0)
		y_tensor = torch.Tensor(y)

		prediction, hidden = rnn(x_tensor, hidden)

		hidden = hidden.data

		loss = criterion(prediction, y_tensor)

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		if batch_i%print_every == 0:
			print('Loss: ', loss.item())
			plt.plot(time_steps[1:], x, 'r.', label='input, x')
			plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.', label='prediction, y')
			plt.legend(loc='best')
			plt.show()
	return rnn

if __name__ == '__main__':
	seq_length = 20
	input_size = 1
	output_size = 1
	hidden_dim = 32
	n_layers = 1

	rnn = RNN(input_size, output_size, hidden_dim, n_layers)

	print(rnn)
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
	
	
	n_steps = 75
	print_every = 25
	trained_rnn = train(rnn, n_steps, print_every)