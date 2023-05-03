from cbc.lm.huggingface_llama_engine import (
    Llama7B,
    Llama13B,
    Llama30B,
    Llama65B,
    Alpaca7B,
    Koala7B,
    Koala13B_V1,
    Koala13B_V2,
    Vicuna_7B,
    Vicuna_13B,
)
import torch


def test_Llama7B_engine() -> None:
    print("test_Llama7B_engine")
    engine = Llama7B()
    completions = engine("The quick brown fox is", n_completions=2)
    assert len(completions) == 2
    for completion in completions:
        print(completion)
    del engine._generator
    del engine.tokenizer
    torch.cuda.empty_cache()

def test_Llama13B_engine() -> None:
    print("test_Llama13B_engine")
    engine = Llama13B()
    completions = engine("The quick brown fox is", n_completions=2)
    assert len(completions) == 2
    for completion in completions:
        print(completion)
    del engine._generator
    del engine.tokenizer
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    test_Llama7B_engine()
    test_Llama13B_engine()