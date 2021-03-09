from transformers import Trainer
from torch.utils import data as D


class PTCTechniqueClassificationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.collate_fn = kwargs["collate_fn"]
        self.batch_siz
        del kwargs["collate_fn"]
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        return D.DataLoader(
            self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.train_batch_size,
        )

    def get_eval_dataloader(self, eval_dataset):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        return D.DataLoader(
            eval_dataset, collate_fn=self.collate_fn, batch_size=self.eval_batch_size
        )

    def get_test_dataloader(self, test_dataset):
        return D.DataLoader(
            test_dataset, collate_fn=self.collate_fn, batch_size=self.eval_batch_size
        )


def collate_fn_with_tokenizer(examples, tokenizer=None):
    return tokenizer.pad(examples, return_tensors="pt")