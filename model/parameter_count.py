from model.config import ModelConfig

def estimate_parameter_count(cfg: ModelConfig) -> dict[str, int]:
    # dimensions of each token
    d = cfg.d_model

    #Token embeddings total
    embedding = cfg.vocab_size * d

    #Attention
    q_proj = d * d
    k_proj = d * cfg.kv_dim
    v_proj = d * cfg.kv_dim
    out_proj = d * d

    attention_per_layer = (
        q_proj + k_proj + v_proj + out_proj
    )

    #SwiGLU:
    #up proj
    #gate
    #down proj
    mlp_per_layer = 3 * d * cfg.d_ff

    # Two RMSNorms per Block
    norms_per_layer = 2 * d

    block_params = (attention_per_layer
                    + mlp_per_layer
                    + norms_per_layer)

    transformer_layers = (cfg.n_layers*block_params)

    # Final RMSNorm
    final_norm = d

    # Extra LM head only when embeddings aren't tied
    lm_head = (
        0
        if cfg.tie_embeddings
        else cfg.vocab_size * d
    )

    total = (
        embedding
        + transformer_layers
        + final_norm
        + lm_head
    )

    return {
        "embedding": embedding,
        "attention_per_layer": attention_per_layer,
        "mlp_per_layer": mlp_per_layer,
        "block_params": block_params,
        "transformer_layers": transformer_layers,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "total": total,
    }

if __name__ == "__main__":
    cfg = ModelConfig()
    count = estimate_parameter_count(cfg)
    for name, value in count.items():
        print(f"{name:25s}: {value/1e6:.3f}M")