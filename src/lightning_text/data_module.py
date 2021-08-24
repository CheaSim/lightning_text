import os
from .base_data_module import BaseDataModule
from .processor import get_dataset, processors
from transformers import AutoTokenizer
from .utils import cache_results, task2file

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class SST2(BaseDataModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.processor = processors[self.args.task_name](self.args.data_dir, self.args.use_prompt)
        self.max_seq_length = self.args.max_seq_length
        
        self.label2id = self.processor.get_labels()
        self.num_labels = len(self.label2id)


    def setup(self, stage=None):
        self.data_train = get_dataset("train", self.args, self.tokenizer, self.processor)
        self.data_val = get_dataset("dev", self.args, self.tokenizer, self.processor)
        self.data_test = get_dataset("test", self.args, self.tokenizer, self.processor)

    def prepare_data(self):
        if not os.path.exists(self.args.data_dir):
            os.system(f"mkdir -vp {self.args.data_dir}")
        file_name = task2file[self.args.task_name]
        os.system(f"wget http://47.97.126.45/dataset/sst-2.tar.gz")
        os.system(f"tar zxvf sst-2.tar.gz")
        os.system("mv SST-2 ./dataset")
        os.system("rm sst-2.tar.gz")


    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="/home/xx/bert-base-uncased", help="Number of examples to operate on per forward step.")
        parser.add_argument("--max_seq_length", type=int, default=128, help="Number of examples to operate on per forward step.")
        return parser

    @cache_results()
    def convert_example_to_feature(self, mode):
        """
        return dataset
        """
        if mode == "train":
            examples = self.processor.get_train_examples(self.data_dir)
        elif mode == "dev":
            examples = self.processor.get_dev_examples(self.data_dir)
        elif mode == "test":
            examples = self.processor.get_test_examples(self.data_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        use_bert = "BertTokenizer" in tokenizer.__class__.__name__  

        for en_idx, example in enumerate(examples):

            inputs = tokenizer(
                example.text_a,
                truncation="longest_first",
                max_length=self.max_seq_length,
                padding="max_length",
                add_special_tokens=True
            )

            if en_idx == 0:
                logger.info(f"input text {example.text_a}")
        
            x = dict()
            x['input_ids'] = inputs['input_ids']
            x['attention_mask'] = inputs['attention_mask']
            if use_bert:
                x['token_type_ids'] = inputs['token_type_ids']
            x['label'] = self.label2id(example)