from metalarena.autodatasets import OmniglotSupervised

from autolightning.dm.few_shot import FewShotMixin


class OmniglotFewShot(FewShotMixin, OmniglotSupervised):
    def get_omniglot_variant(self):
        return super().get_omniglot_variant("few-shot")

    def get_dataset(self, phase: str):
        if phase == 'train':
            return self.train_set
        elif phase == 'val':
            return self.val_set
        elif phase == 'test':
            return self.test_set
        
        raise ValueError(f"Unsupported phase: {phase}")
    
    def setup(self, stage: str):
        (train_set, val_set, test_set), _ = self.get_omniglot_variant()

        if stage == 'fit':
            self.train_set = train_set
            self.val_set = val_set
        elif stage == 'validate':
            self.val_set = val_set
        elif stage == 'test':
            self.test_set = test_set
        else:
            raise ValueError(f"Unsupported stage: {stage}")
