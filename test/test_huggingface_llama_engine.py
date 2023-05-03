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


def test_Llama30B_engine() -> None:
    print("test_Llama30B_engine")
    engine = Llama30B()
    completions = engine("The quick brown fox is", n_completions=2)
    assert len(completions) == 2
    for completion in completions:
        print(completion)
    del engine
    torch.cuda.empty_cache()


def test_Llama65B_engine() -> None:
    print("test_Llama65B_engine")
    engine = Llama65B()
    completions = engine("The quick brown fox is", n_completions=2)
    assert len(completions) == 2
    for completion in completions:
        print(completion)
    del engine
    torch.cuda.empty_cache()


def test_Alpaca7B_engine() -> None:
    print("test_Alpaca7B_engine")
    engine = Alpaca7B()
    completions = engine("The quick brown fox is", n_completions=2)
    assert len(completions) == 2
    for completion in completions:
        print(completion)
    del engine
    torch.cuda.empty_cache()


def test_Koala7B_engine() -> None:
    print("test_Koala7B_engine")
    engine = Koala7B()
    completions = engine("The quick brown fox is", n_completions=2)
    assert len(completions) == 2
    for completion in completions:
        print(completion)
    del engine
    torch.cuda.empty_cache()


def test_Koala13B_V1_engine() -> None:
    print("test_Koala13B_V1_engine")
    engine = Koala13B_V1()
    completions = engine("The quick brown fox is", n_completions=2)
    assert len(completions) == 2
    for completion in completions:
        print(completion)
    del engine
    torch.cuda.empty_cache()


def test_Koala13B_V2_engine() -> None:
    print("test_Koala13B_V2_engine")
    engine = Koala13B_V2()
    completions = engine("The quick brown fox is", n_completions=2)
    assert len(completions) == 2
    for completion in completions:
        print(completion)
    del engine
    torch.cuda.empty_cache()


def test_Vicuna_7B_engine() -> None:
    print("test_Vicuna_7B_engine")
    engine = Vicuna_7B()
    completions = engine("The quick brown fox is", n_completions=2)
    assert len(completions) == 2
    for completion in completions:
        print(completion)
    del engine
    torch.cuda.empty_cache()


def test_Vicuna_13B_engine() -> None:
    print("test_Vicuna_13B_engine")
    engine = Vicuna_13B()
    completions = engine("The quick brown fox is", n_completions=2)
    assert len(completions) == 2
    for completion in completions:
        print(completion)
    del engine
    torch.cuda.empty_cache()


def test_Llama7B_summary_engine() -> None:
    print("test_Llama7B_summary_engine")
    engine = Llama7B()
    summary = engine.best("The quick brown fox is")
    print(summary)
    del engine
    torch.cuda.empty_cache()


def test_Llama13B_summary_engine() -> None:
    print("test_Llama13B_summary_engine")
    engine = Llama13B()
    summary = engine.best("The quick brown fox is")
    print(summary)
    del engine
    torch.cuda.empty_cache()


def test_Llama30B_summary_engine() -> None:
    print("test_Llama30B_summary_engine")
    engine = Llama30B()
    summary = engine.best("The quick brown fox is")
    print(summary)
    del engine
    torch.cuda.empty_cache()


def test_Llama65B_summary_engine() -> None:
    print("test_Llama65B_summary_engine")
    engine = Llama65B()
    summary = engine.best("The quick brown fox is")
    print(summary)
    del engine
    torch.cuda.empty_cache()


def test_Alpaca7B_summary_engine() -> None:
    print("test_Alpaca7B_summary_engine")
    engine = Alpaca7B()
    summary = engine.best("The quick brown fox is")
    print(summary)
    del engine
    torch.cuda.empty_cache()


def test_Koala7B_summary_engine() -> None:
    print("test_Koala7B_summary_engine")
    engine = Koala7B()
    summary = engine.best("The quick brown fox is")
    print(summary)
    del engine
    torch.cuda.empty_cache()


def test_Koala13B_V1_summary_engine() -> None:
    print("test_Koala13B_V1_summary_engine")
    engine = Koala13B_V1()
    summary = engine.best("The quick brown fox is")
    print(summary)
    del engine
    torch.cuda.empty_cache()


def test_Koala13B_V2_summary_engine() -> None:
    print("test_Koala13B_V2_summary_engine")
    engine = Koala13B_V2()
    summary = engine.best("The quick brown fox is")
    print(summary)
    del engine
    torch.cuda.empty_cache()


def test_Vicuna_7B_summary_engine() -> None:
    print("test_Vicuna_7B_summary_engine")
    engine = Vicuna_7B()
    summary = engine.best("The quick brown fox is")
    print(summary)
    del engine
    torch.cuda.empty_cache()


def test_Vicuna_13B_summary_engine() -> None:
    print("test_Vicuna_13B_summary_engine")
    engine = Vicuna_13B()
    summary = engine.best("The quick brown fox is")
    print(summary)
    del engine
    torch.cuda.empty_cache()

    
if __name__ == "__main__":
    test_Llama7B_engine()
    test_Llama13B_engine()
    # test_Llama30B_engine()
    # test_Llama65B_engine()
    # test_Alpaca7B_engine()
    # test_Koala7B_engine()
    test_Koala13B_V1_engine()
    test_Koala13B_V2_engine()
    # test_Vicuna_7B_engine()
    # test_Vicuna_13B_engine()
    # test_Llama7B_summary_engine()
    # test_Llama13B_summary_engine()
    # test_Llama30B_summary_engine()
    # test_Llama65B_summary_engine()
    # test_Alpaca7B_summary_engine()
    # test_Koala7B_summary_engine()
    # test_Koala13B_V1_summary_engine()
    # test_Koala13B_V2_summary_engine()
    # test_Vicuna_7B_summary_engine()
    # test_Vicuna_13B_summary_engine()
