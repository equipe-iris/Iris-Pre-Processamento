from transformers import BartTokenizer, BartForConditionalGeneration
import os
import math
from datasets import Dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

def summarization_training(df):
    # 1) Extrai listas de inputs e labels
    texts    = df['Interacao'].astype(str).tolist()
    summaries = df['Resumo'].astype(str).tolist()

    # 2) Cria um Dataset HF e faz split treino/val (90/10)
    ds = Dataset.from_dict({'text': texts, 'summary': summaries})
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split['train']
    eval_ds  = split['test']

    # 4) Carrega tokenizer e modelo originais
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model     = BartForConditionalGeneration.from_pretrained(model_name)

    # 5) Tokenização + auto-encoding (labels = input_ids)
    def preprocess(examples):
        inputs = tokenizer(
            examples['text'],
            max_length=1024,
            truncation=True,
        )
        # tokeniza o resumo alvo
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                examples['summary'],
                max_length=128,
                truncation=True,
            )
        inputs['labels'] = targets['input_ids']
        return inputs

    tokenized_train = train_ds.map(
        preprocess,
        batched=True,
        remove_columns=['text', 'summary']
    )
    tokenized_eval = eval_ds.map(
        preprocess,
        batched=True,
        remove_columns=['text', 'summary']
    )

    # 6) Configurações de treino
    base_dir   = os.path.dirname(__file__)
    app_dir    = os.path.abspath(os.path.join(base_dir, '..'))
    output_dir = os.path.join(app_dir, 'artifacts', 'summarization')
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        do_eval=True,
        eval_steps=500,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        fp16=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 7) Roda o pré-treino
    train_result = trainer.train()
    metrics_train = train_result.metrics

    # 8) Avaliação
    metrics_eval = trainer.evaluate()
    eval_loss    = metrics_eval['eval_loss']
    perplexity   = math.exp(eval_loss)

    # 9) Salva modelo/tokenizer adaptados
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 10) Retorna métricas básicas
    return {
        'message':     'Fine-tuning supervisionado concluído.',
        'model_dir':   output_dir,
        'train_loss':  metrics_train.get('loss'),
        'eval_loss':   eval_loss,
        'perplexity':  perplexity,
        'epochs':      training_args.num_train_epochs
    }