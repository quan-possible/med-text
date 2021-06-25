from datamodule import DataModule, Collator
from tokenizer import Tokenizer
from pytorch_lightning import seed_everything

def test_collator(collator):
    
    # nice = {'text':'Ghrelin was identified in the stomach as an endogenous\
    # ligand specific for the growth hormone secretagogue receptor\
    # ( GHS-R ) .\nGHS-R is found in various tissues , but its function\
    # is unknown .\nHere we show that GHS-R is found in hepatoma cells\
    # .\nExposure of these cells to ghrelin caused up-regulation of\
    # several insulin-induced activities including tyrosine phosphorylation\
    # of insulin receptor substrate-1 ( IRS-1 ) , association of the\
    # adapter molecule growth factor receptor-bound protein 2 with IRS-1\
    # , mitogen-activated protein kinase activity , and cell proliferation\
    # .\nUnlike insulin , ghrelin inhibited Akt kinase activity as well as\
    # up-regulated gluconeogenesis .\nThese findings raise the possibility\
    # that ghrelin modulates insulin activities in humans .\n', 'labels':[33]}
    
    

    MODEL = "bert-base-uncased"
    DATA_PATH = "./project/data"
    FILE_PATH = "./tests/hoc_sample.csv"
    DATASET = "hoc"
    BATCH_SIZE = 2
    NUM_WORKERS = 2
    
    tokenizer = Tokenizer(MODEL)
    collator = Collator(tokenizer)
    # datamodule = DataModule(
    #     tokenizer, collator, DATA_PATH,
    #     DATASET, BATCH_SIZE, NUM_WORKERS
    # )
    
    dicts = DataModule.read_csv(FILE_PATH)
    
    out_inp, out_targets = collator(dicts)
    
    
    

