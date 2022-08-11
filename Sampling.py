class Sampling:

	#every sampling algorithm should provide the indicies of the data points to be selected 
	def __init__(self, train_logits, top_k):
		self.top_k=top_k
		self.train_logits=train_logits 

	def random_samp(self):

		logits = torch.tensor(self.train_logits)
		categorical = Categorical(logits = logits)
		scores = categorical.entropy()
		_, top_unsampled = torch.topk(scores, self.top_k)
	
		return top_unsampled

	def entropy_samp(self):

		logits = torch.tensor(self.train_logits)
		categorical = Categorical(logits = logits)
		scores = categorical.entropy()
		_, top_unsampled = torch.topk(scores, self.top_k)
	
		return top_unsampled

	def sampling_call(self, choose_samp):
		if choose_samp=='random_samp':
			top_unsampled=self.random_samp()
		elif choose_samp=='entropy_samp':
			top_unsampled=self.entropy_samp()
		return top_unsampled
