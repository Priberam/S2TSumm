import pytorch_lightning as pl

class ParentModule(pl.LightningModule):
    def __init__(self):
        super().__init__()