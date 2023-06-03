from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.models.parent import ParentModule
from src.data.data_module import S2TSumDataMod


class ASRLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):  

        parser.link_arguments('data.ctc_vocab_file', 'model.init_args.ctc_vocab_file', apply_on='parse')
        parser.link_arguments('data.ctc_delimeter', 'model.init_args.ctc_delimeter', apply_on='parse')
        parser.link_arguments('model.init_args.batch_size', 'data.batch_size', apply_on='parse')
        parser.link_arguments('model.accumulate_grad_batches', 'trainer.accumulate_grad_batches', apply_on='parse')

        parser.add_lightning_class_args(EarlyStopping, 'early_stopping')
        parser.set_defaults({
            'early_stopping.monitor': 'metrics/wer',
            'early_stopping.mode': 'min',
            'early_stopping.patience': 20
        })

        parser.add_lightning_class_args(ModelCheckpoint, 'val_ctc_loss_checkpoint')
        parser.set_defaults({
            'val_ctc_loss_checkpoint.monitor': 'val/ctc_loss',
            'val_ctc_loss_checkpoint.mode': 'min',
            'val_ctc_loss_checkpoint.save_top_k': 3,
            'val_ctc_loss_checkpoint.save_last': True,
            'val_ctc_loss_checkpoint.filename': '{epoch}-{step}'
        })

        parser.add_lightning_class_args(ModelCheckpoint, 'wer_checkpoint')
        parser.set_defaults({
            'wer_checkpoint.monitor': 'metrics/wer',
            'wer_checkpoint.mode': 'min',
            'wer_checkpoint.save_top_k': 3,
            'wer_checkpoint.save_last': True,
            'wer_checkpoint.filename': 'wer-{epoch}-{step}'
        })

        parser.add_lightning_class_args(ModelSummary, 'model_summary')
        parser.set_defaults({'model_summary.max_depth': -1})


def cli_main():
    cli = ASRLightningCLI(
        ParentModule,
        S2TSumDataMod,
        subclass_mode_model = True,
        save_config_overwrite = True
    )


if __name__ == '__main__':
    
    cli_main()