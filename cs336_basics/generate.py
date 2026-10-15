import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer.transformer import TransformerLM
from cs336_basics.training.config import load_config

# Model config - match training
config = load_config("configs/default_config.yaml")
cfg = config.model

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = Tokenizer.from_files(
        vocab_path="data/tinystories_vocab.json",
        merges_path="data/tinystories_merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    eos_token_id = tokenizer.inverse_vocab[b"<|endoftext|>"]
        
    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        num_layers=cfg.num_layers,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        theta=cfg.theta,
    )

    checkpoint = torch.load("./checkpoints/checkpoint-epoch0-step5000-final.pt", map_location="cpu")
    # Strip "_orig_mod." prefix from keys (from torch.compile)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {device}")
    print("-" * 50)

    prompt = "Once upon a time"
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

    # Generate with different settings
    for temp, top_p in [(0.7, 0.9), (1.0, 1.0), (0.0, 1.0)]:
        print(f"\n=== Temperature: {temp}, Top-p: {top_p} ===")

        output_ids = model.generate(
            input_ids=input_ids,
            max_tokens=256,
            temperature=temp,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

        generated_text = tokenizer.decode(output_ids[0].tolist())
        print(generated_text)
        print(f"\nTokens generated: {len(output_ids[0])}")

if __name__ == "__main__":
    main()
