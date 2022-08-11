class LanguageModel:
    
    def __init__(self, checkpoint="distilbert-base-uncased"):

        ##Define variables that doesn't depend on data, batch size, or sampling
        ##checkpoint name is needed for all the methods
        self.klabels=2
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
  

    def data_loader(self, dataset_name, collection):

        ##Handle loading independent and part of collection datasets
        if collection == None:
            raw_datasets = load_dataset(dataset_name)
        else:
            raw_datasets = load_dataset(collection, dataset_name)

        return raw_datasets


    def data_tokenizer(self,example):
        toenization = self.tokenizer(example["sentence"], truncation=True)  

        return toenization
        
    def label_counter(self, tokenized_datasets):

        return len(set(tokenized_datasets['train']['label']))
         

    def data_trainer(self, train_data, test_data, klabels=2):

        default_args = {

            "output_dir": "tmp",
            "num_train_epochs": 1}

        training_args = TrainingArguments(per_device_train_batch_size=8, **default_args)
        model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=klabels).to("cuda")

        #for first iteration 

        trainer = Trainer(
            model,
            training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer)

        return trainer 


    def training(self, queries_unsampled, dataset_name, collection=None):

        raw_datasets = self.data_loader(dataset_name, collection)
        tokenized_datasets = raw_datasets.map(self.data_tokenizer, batched=True)
        self.klabels = self.label_counter(tokenized_datasets)
        if len(queries_unsampled)==0: 
          print('first iterationnnnnnnnnnnnnnnnnnnnnnn')
          train_idxs = [i for i in range(0, len(tokenized_datasets['train']))]
          queries_unsampled = random.sample(train_idxs, 10)
          train_dataset = Dataset.from_dict(tokenized_datasets['train'][queries_unsampled])
        else:
          print('following iterationnnnnnnnnnnnnnnnnnnnnnn')
          train_dataset = Dataset.from_dict(tokenized_datasets['train'][queries_unsampled])
        test_dataset = tokenized_datasets['validation']
        trainer = self.data_trainer(train_dataset, test_dataset, self.klabels)
        trainer.train()
        ## to choose the next points 
        ## Delete used data points first 
        print('ohhhhhhhhhhhhhhhhhhhhhhhhh', queries_unsampled)
        pool_idx = np.delete(torch.arange(len(raw_datasets['train'])), queries_unsampled)
        train_pool = Dataset.from_dict(tokenized_datasets["train"][pool_idx])
        train_logits = trainer.predict(train_pool)[0]
        test_logits,test_labels, _ = trainer.predict(tokenized_datasets["validation"])

        return train_logits, test_logits, test_labels, train_pool

    def evaluating(self, logits, test_labels):

        preds = np.argmax(logits, axis=-1)
        f1_metric = load_metric("f1")
        results = f1_metric.compute(predictions=preds, references=test_labels, average="micro")
        return results
