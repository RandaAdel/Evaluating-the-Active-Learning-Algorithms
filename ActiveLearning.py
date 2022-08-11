def ActiveLearning(top_k, dataset_name, collection=None, choose_samp = 'entropy_samp', checkpoint="distilbert-base-uncased"):
  queries_unsampled = []
  results_f1 = []
  language_model = LanguageModel(checkpoint)

  # first iteration send raw_dataset
  print("we got hereeeeeeeeeeeeeeeeeeeeeeeee")
  train_logits, test_logits, test_labels, train_pool = language_model.training(queries_unsampled, dataset_name, collection)
  print('sucesssssssssssssssssssssssssssssssssssssss')
  # zero labeled data iteration using the pretrained model without fine tuning 
  results = language_model.evaluating(test_logits, test_labels)['f1']
  results_f1.append(results)

  for repeat in range(0, 2):
    Sampling_method = Sampling(train_logits, top_k)
    top_unsampled = Sampling_method.sampling_call(choose_samp)
    queries_unsampled = queries_unsampled + top_unsampled.tolist()
    train_logits, test_logits, test_labels, train_pool = language_model.training(queries_unsampled, dataset_name, collection)
    results = language_model.evaluating(test_logits, test_labels)['f1']
    results_f1.append(results)
    print('f1 for {} iterations is: '.format(repeat), results)

  return results_f1, queries_unsampled

def plot_results(dataset_name, top_k, sampling_f1, sampling_method):
  ls_=[x for x in range(top_k, top_k*len(sampling_f1), top_k)]
  print(ls_)
  ls_.insert(0, 10)
  ls_.insert(0, 0)
  sampling_f1.insert(0, 0)
  plt.plot(ls_,sampling_f1, label = sampling_method)
  