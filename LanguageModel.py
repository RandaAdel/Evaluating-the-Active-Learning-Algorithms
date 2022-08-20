from transformers import TrainingArguments, Trainer, logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TextClassificationPipeline
from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader

class LanguageModel:
    
    def __init__(self, checkpoint="distilbert-base-uncased"):

        ##Define variables that doesn't depend on data, batch size, or sampling
        ##checkpoint name is needed for all the methods
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
  

    def data_loader(self, dataset_name, collection):

        ##Handle loading independent and part of collection datasets
        if collection == None:
            raw_datasets = load_dataset(dataset_name)
        else:
            raw_datasets = load_dataset(collection, dataset_name)

        txt_sent=[]
        for i in raw_datasets['train'].features:
          if i in ['text', 'sentence', 'sentence1', 'sentence2']:
            txt_sent.append(i) 

          if i in ['label','labels']:
            label_col = i

        ##See which to use validation or test set
        if(len(set(raw_datasets['test']['label']))>=2):
          test_name ='test'
        else:
          test_name ='validation'


        return raw_datasets, txt_sent, label_col, test_name

    def data_tokenizer_one(self,example):
        toenization = self.tokenizer(example["sentence"], truncation=True)  
      
    def data_tokenizer_two(self,example):
        toenization = self.tokenizer(example["text"], truncation=True)  

        return toenization

    def data_tokenizer_three(self,example):
        toenization = self.tokenizer(example["sentence1"], example["sentence2"], truncation=True)  

        return toenization
        
    def label_counter(self, tokenized_datasets, label):
        #handle label and labels 
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


    def training(self, train_dataset, test_dataset, unlabeled_pool, klabels):

        trainer = self.data_trainer(train_dataset, test_dataset, klabels)
        trainer.train()
        train_logits = trainer.predict(unlabeled_pool)[0]
        test_logits,test_labels, _ = trainer.predict(test_dataset)

        return train_logits, test_logits, test_labels

    def evaluating(self, logits, test_labels):

        preds = np.argmax(logits, axis=-1)
        f1_metric = load_metric("f1")
        results = f1_metric.compute(predictions=preds, references=test_labels, average="micro")
        return results
