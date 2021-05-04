from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
from datasets import load_metric, load_dataset

from IPython import embed
import pandas as pd
import numpy as np
import nltk

#Adapted from https://github.com/huggingface/notebooks/blob/master/examples/summarization.ipynb

def main():
    dataset = load_dataset('csv', data_files="../data/uqv100_pairs.csv")
    model_checkpoint = "t5-base" #["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    max_input_length = 128
    max_target_length = 64

    col_from = "query_x"
    col_to = "query_y"

    def preprocess_function(examples):
        prefix = "paraphrase: "
        inputs = [prefix + doc for doc in examples[col_from]]        
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer([t for t in examples[col_to]], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 32
    args = Seq2SeqTrainingArguments(
        "paraphrase-fine-tuned-uqv",
        # evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True
        # fp16=True,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets['train'],        
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    print("Fine-tuning T5.")
    trainer.train()
    model.save_pretrained("../data/{}_uqv_variation_paraphraser".format(model_checkpoint))

if __name__ == "__main__":
    main()