from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

def train_bpe_tokenizer(files: list[str], vocab_size: int = 32000, save_path: str = "tokenizer/bpe_tokenizer.json"):
    tokenizer = HFTokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder() 

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )

    tokenizer.train(files, trainer)
    tokenizer.save(save_path)
    return tokenizer