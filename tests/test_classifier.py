import sys
sys.path.insert(0, 'D:/OneDrive - Aalto University/Courses/Thesis/med-text')
# TODO: Find a better way.

import torch

from pytorch_lightning import seed_everything
from project.classifier import Classifier
from project.datamodule import DataModule, Collator
from project.tokenizer import Tokenizer


class TestModel():
    
    def __init__(self, model: Classifier):
        self.model = model
        self.optim = model.configure_optimizers()[0][0]
        
    def test_shape(self, batch, batch_size, n_classes):
        inputs, targets = batch
        self.model.eval()
        res = self.model(inputs)

        err_msg = "Incorrect output shape!"
        assert res['logits'].size() == torch.Size(
            [batch_size, n_classes]), err_msg
        
    def _training_step(self, batch):
            # put model in train mode
        self.model.train()
        # run one forward + backward step
        # clear gradient
        self.optim.zero_grad()
        # inputs and targets
        inputs, targets = batch
        # forward
        likelihood = self.model.forward(inputs)
        # calc loss
        loss = self.model.loss(likelihood, targets)
        # backward
        loss.backward()
        # optimization step
        self.optim.step()
        
    def test_params_update(self, batch, vars_change=True, params=None):
        """Check if given variables (params) change or not during training
        If parameters (params) aren't provided, check all parameters.
        Parameters
        ----------
        vars_change : bool
        a flag which controls the check for change or not change
        batch : list
        a 2 element list of inputs and labels, to be fed to the model
        params : list, optional
        list of parameters of form (name, variable)
        Raises
        ------
        VariablesChangeException
        if vars_change is True and params DO NOT change during training
        if vars_change is False and params DO change during training
        """

        if params is None:
            # get a list of params that are allowed to change
            params = [np for np in model.named_parameters()
                      if np[1].requires_grad]

        # take a copy
        initial_params = [(name, p.clone()) for (name, p) in params]

        # run a training step
        self._training_step(batch)

        # check if variables have changed
        for (_, p0), (name, p1) in zip(initial_params, params):
            try:
                if vars_change:
                    assert not torch.equal(p0, p1)
                else:
                    assert torch.equal(p0, p1)
            except AssertionError:
                raise Exception(f"{name} {'did not change!' if vars_change else 'changed!'}")

# class TestClassifier():
    
#     def __init__(self, batch_size=2,
#                  encoder_model="bert-base-uncased",
#                  data_path="./project/data",
#                  dataset="hoc",
#                  num_workers=2,
#                  nr_frozen_epochs=1,
#                  encoder_learning_rate=1e-05,
#                  learning_rate=3e-05,
#                  random_sampling=False,
#                  device='cpu') -> None:
        
#         self.batch_size = batch_size
#         self.tokenizer = Tokenizer(encoder_model)
#         self.collator = Collator(self.tokenizer)
        
#         self.datamodule = DataModule(
#             self.tokenizer, self.collator, data_path,
#             dataset, self.batch_size, num_workers,
#             rand_sampling=random_sampling,
#         )

#         self.n_classes = self.datamodule.n_classes

#         self.model = Classifier(
#             self.tokenizer, self.collator, encoder_model,
#             self.batch_size, self.n_classes, nr_frozen_epochs,
#             encoder_learning_rate, learning_rate,
#         )
        
#         self.optim = self.model.configure_optimizers()[0][0]
        
#     def _training_step(self, model, batch):
#         # put model in train mode
#         model.train()

#         # run one forward + backward step
#         # clear gradient
#         self.optim.zero_grad()
#         # inputs and targets
#         inputs, targets = batch
#         # forward
#         likelihood = self.model(inputs)
#         # calc loss
#         loss = self.model.loss(likelihood, targets)
#         # backward
#         loss.backward()
#         # optimization step
#         self.optim.step()

#     def test_shape(self, batch):
        
#         inputs, targets = batch
#         res = self.model(inputs)

#         err_msg = "Incorrect output shape!"
#         assert res['logits'].size() == torch.Size(
#             [self.batch_size, self.n_classes]), err_msg
    

    

if __name__ == "__main__":
    
    BATCH_SIZE = 2
    ENCODER_MODEL = "bert-base-uncased"
    DATA_PATH = "./project/data"
    DATASET = "hoc"
    NUM_WORKERS = 2
    NR_FROZEN_EPOCHS = 1
    ENCODER_LEARNING_RATE = 1e-05
    LEARNING_RATE = 3e-05
    RANDOM_SAMPLING = False

    seed_everything(69)

    tokenizer = Tokenizer(ENCODER_MODEL)
    collator = Collator(tokenizer)
    datamodule = DataModule(
        tokenizer, collator, DATA_PATH,
        DATASET, BATCH_SIZE, NUM_WORKERS,
        rand_sampling=RANDOM_SAMPLING,
    )

    n_classes = datamodule.n_classes

    model = Classifier(
        tokenizer, collator, ENCODER_MODEL,
        BATCH_SIZE, n_classes, NR_FROZEN_EPOCHS,
        ENCODER_LEARNING_RATE, LEARNING_RATE,
    )
    
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    print(batch)
    
    test_obj = TestModel(model)
    test_obj.test_shape(batch, BATCH_SIZE, n_classes)
    test_obj.test_params_update(batch)
    
    print("Test successful!")
