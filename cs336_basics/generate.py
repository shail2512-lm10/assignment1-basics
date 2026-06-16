import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer.transformer import TransformerLM
from cs336_basics.training.config import load_config

# Model config - match training
config = load_config("configs/default_config.json")
cfg = config.model

def main():
    device = "mps"

    tokenizer = Tokenizer.from_files(
        vocab_filepath="results/TinyStoriesV2_GPT4_train_vocab.pkl",
        merges_filepath="results/TinyStoriesV2_GPT4_train_merges.pkl",
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
        device=device,
    )

    checkpoint = torch.load("./checkpoints/checkpoint_final.pt", map_location=device)
    # Strip "_orig_mod." prefix from keys (from torch.compile)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(state_dict)
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
            input_ids,
            max_new_tokens=256,
            temperature=temp,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

        generated_text = tokenizer.decode(output_ids[0].tolist())
        print(generated_text)
        print(f"\nTokens generated: {output_ids.shape[1]}")

if __name__ == "__main__":
    main()