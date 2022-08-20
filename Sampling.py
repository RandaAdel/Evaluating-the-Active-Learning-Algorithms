class Sampling:

  #every sampling algorithm should provide the indicies of the data points to be selected 
  def __init__(self, train_logits, train_dataset, unlabeled_pool, top_k):
    self.top_k=top_k
    self.train_logits=train_logits 
    self.train_dataset=train_dataset
    self.unlabeled_pool=unlabeled_pool

  def random_samp(self):

    train_idxs = [i for i in range(0, len(self.unlabeled_pool))]
    top_unsampled = random.sample(train_idxs, self.top_k)
  
    return top_unsampled

  def entropy_samp(self):
    #print(self.train_logits)
    logits = torch.tensor(self.train_logits)
    categorical = Categorical(logits = logits)
    scores = categorical.entropy()
    _, top_unsampled = torch.topk(scores, self.top_k)

    return top_unsampled

  def sampling_call(self, choose_samp):
    
    if choose_samp=='random':
      top_unsampled=self.random_samp()
    elif choose_samp=='entropy':
      top_unsampled=self.entropy_samp()
    
    #Subset train  
    new_subset = self.unlabeled_pool[top_unsampled]
    train_dic = self.train_dataset[:]
    for key, value in new_subset.items():
      train_dic[key]=train_dic[key]+new_subset[key]

    train_subset = Dataset.from_dict(train_dic)
    #Update UnlabeledPool
    pool_idx = np.delete(torch.arange(len(self.unlabeled_pool)), top_unsampled)
    Updated_pool = Dataset.from_dict(self.unlabeled_pool[pool_idx])

    return train_subset, Updated_pool
