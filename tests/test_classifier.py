import sys
sys.path.insert(0, 'project')
# TODO: Find a better way.

import torch

from pytorch_lightning import seed_everything
from classifier import Classifier
from datamodule import DataModule, Collator
from tokenizer import Tokenizer


class TestModel():
    
    def __init__(self, model: Classifier):
        self.model = model
        self.optim = model.configure_optimizers()[0][0]
        
    def test_shape(self, batch, batch_size, n_classes):
        inputs, targets = batch
        
        self.model.eval()
        res = self.model(inputs)['logits']

        err_msg = "Incorrect output shape!"
        assert res.size() == torch.Size(
            [batch_size, n_classes]), err_msg
        
        print("Shapes look good!")
        
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
        
    def test_params_update(self, batch, exceptions=["bert.pooler.dense.weight",
                                                    "bert.pooler.dense.bias"], 
                           vars_change=True, params=None):
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
        # for (name,_) in initial_params:
        #     print(name)

        # run a training step
        self._training_step(batch)

        # check if variables have changed
        suspects = []
        for (_, p0), (name, p1) in zip(initial_params, params):
            
            if name not in exceptions:
                if (vars_change and torch.equal(p0, p1)) \
                    or (not vars_change and not torch.equal(p0, p1)):
                    suspects.append(name)

        err_msg = f"{suspects}{' did not change!' if vars_change else 'changed!'} after optimization."
        assert len(suspects) == 0, err_msg
        
        print("Parameters changed after optimization!")
    

if __name__ == "__main__":
    
    BATCH_SIZE = 2
    ENCODER_MODEL = "bert-base-cased"
    DATA_PATH = "./project/data"
    DATASET = "mtc"
    NUM_WORKERS = 2
    NR_FROZEN_EPOCHS = 0
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
    # print(batch)
    
    test_obj = TestModel(model)
    test_obj.test_shape(batch, BATCH_SIZE, n_classes)
    test_obj.test_params_update(batch)
    
    print("Test successful!")
