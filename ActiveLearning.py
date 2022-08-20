import matplotlib.pyplot as plt

class Analysis:
  
  def __init__(self, top_k, dataset_name, checkpoint="distilbert-base-uncased"):
      self.top_k=top_k
      self.dataset_name=dataset_name
      self.checkpoint=checkpoint
  
  def ActiveLearning(self, choose_samp, collection=None):
    queries_unsampled = []
    results_f1 = []
    language_model = LanguageModel(self.checkpoint)
    raw_datasets, txt_sent, label_col, test_name = language_model.data_loader(self.dataset_name, collection)
    #To deal with both datasets
    if (len(txt_sent)==1 and txt_sent[0]=='sentence'):
      print('token oneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
      tokenized_datasets = raw_datasets.map(language_model.data_tokenizer_one, batched=True)
    elif (len(txt_sent)==1 and txt_sent[0]=='text'):
      print('token twoooooooooooooooooooooooooooooooooo')
      tokenized_datasets = raw_datasets.map(language_model.data_tokenizer_two, batched=True)
    else:
      print('token threeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
      tokenized_datasets = raw_datasets.map(language_model.data_tokenizer_three, batched=True)
    klabels = language_model.label_counter(tokenized_datasets, txt_sent[0])
    # small subset
    train_dataset = Dataset.from_dict(tokenized_datasets['train'][0:2])
    # unlabeled pool
    unlabeled_pool = tokenized_datasets['train']
    test_dataset = tokenized_datasets[test_name]


    # first iteration send raw_dataset
    print("we got hereeeeeeeeeeeeeeeeeeeeeeeee")
    train_logits, test_logits, test_labels = language_model.training(train_dataset, test_dataset, unlabeled_pool, klabels)
    # zero labeled data iteration using the pretrained model without fine tuning 
    results = language_model.evaluating(test_logits, test_labels)['f1']
    results_f1.append(results)

    for repeat in range(0, 2):
      Sampling_method = Sampling(train_logits, train_dataset, unlabeled_pool, self.top_k)
      train_dataset, unlabeled_pool = Sampling_method.sampling_call(choose_samp)
      train_logits, test_logits, test_labels = language_model.training(train_dataset, test_dataset, unlabeled_pool, klabels)
      results = language_model.evaluating(test_logits, test_labels)['f1']
      results_f1.append(results)
      print('f1 for {} iterations is: '.format(repeat), results)

    return results_f1, queries_unsampled

  def plot_results(self, sampling_f1, choose_samp):
    ls_=[x for x in range(self.top_k, self.top_k*len(sampling_f1), self.top_k)]
    #print(ls_)
    ls_.insert(0, 2)
    ls_.insert(0, 0)
    sampling_f1.insert(0, 0)
    plt.plot(ls_,sampling_f1, label = choose_samp)

  def analysis_call(self, collection=None):
    random.seed(10)
    entropy_f1, queries_unsampled = self.ActiveLearning('entropy', collection)
    random_f1, queries_unsampled = self.ActiveLearning('random', collection)
    self.plot_results(entropy_f1, 'entropy')
    self.plot_results(random_f1, 'random')

    plt.xlabel('number of points used for training the model')
    plt.ylabel('f1 score for the test data')
    plt.title(("Evaluation of entropy using {} data set").format(self.dataset_name))
    plt.legend(loc='lower right')     
    plt.show()

    
  
