# H-GRPO Training Loop & Gradient Flow (TurnRD as decomposer)

Visual reference for the H-GRPO PPO-style trainer in
`src/algorithms/grpo/trainer.py` with the **Method B** TurnRD decomposer
(`src/algorithms/hgpo/decomposers/turnrd.py`) wired in.

---

## 1. Top-level training loop (rollout ↔ trainer ↔ vLLM)

```mermaid
flowchart TB
    subgraph Rollout["Rollout server (vLLM) — policy snapshot pi_old"]
        ENV["WebShop env<br/>WebShopAdapter<br/>state s_t, valid_actions"]
        LLMR["LoRA-adapted Qwen<br/>(serving)"]
        ROLL["Collector → TrajectoryGroup<br/>K trajectories per task<br/>store: prompt_ids, action_ids,<br/>action_token_logprobs, R_i"]
    end

    subgraph Trainer["HGPOTrainer (Modal A100)"]
        DEC["TurnRD Decomposer<br/>alpha_t per turn → r_hat_t = alpha_t · R"]
        ADV["Advantages<br/>A_traj, A_turn, A_H"]
        FWD_NEW["Fwd: pi_theta (grad on)<br/>_batched_logprobs use_ref=False"]
        FWD_REF["Fwd: pi_ref (no_grad)<br/>disable_adapter()<br/>or SFT-LoRA snapshot"]
        LOSS["L = L_PPO + beta·KL + lambda·L_cons"]
        BWD["loss.backward()<br/>scaled by 1/grad_accum"]
        OPT1["AdamW (LoRA)<br/>lr=1e-6"]
        OPT2["AdamW (TurnRD)<br/>lr=1e-4"]
        KLC["AdaptiveKLController<br/>updates beta"]
    end

    SYNC["weight_sync<br/>push LoRA → vLLM"]

    ENV -->|"obs_t"| LLMR
    LLMR -->|"a_t (text + tokens + logp)"| ENV
    LLMR --> ROLL
    ENV --> ROLL

    ROLL -->|"TrajectoryGroup"| DEC
    DEC -->|"per-turn rewards r_hat_t"| ADV
    ADV -->|"A_H broadcast over tokens"| LOSS

    ROLL -->|"prompt_ids, action_ids"| FWD_NEW
    ROLL -->|"prompt_ids, action_ids"| FWD_REF
    FWD_NEW -->|"new logp pi_theta"| LOSS
    FWD_REF -->|"ref logp pi_ref"| LOSS
    ROLL -->|"old logp pi_old (constants)"| LOSS

    LOSS --> BWD
    BWD -->|"grad LoRA params"| OPT1
    BWD -->|"grad TurnRD params"| OPT2
    LOSS -->|"observed_kl"| KLC
    KLC -->|"beta_{t+1}"| LOSS

    OPT1 -->|"updated LoRA theta"| SYNC
    SYNC -->|"swap weights"| LLMR
```

---

## 2. Gradient flow — where ∂L/∂θ goes

```mermaid
flowchart LR
    subgraph Inputs["Constants (no grad)"]
        OLD["log pi_old<br/>(rollout cache)"]
        REF["log pi_ref<br/>(disable_adapter, no_grad)"]
        R["R_i (final rewards)"]
    end

    subgraph PolicyParams["LoRA params theta_LoRA<br/>(only trainable LLM weights)"]
        LORA["A, B matrices<br/>on attention proj layers"]
    end

    subgraph TurnRDParams["TurnRD params phi"]
        CLS["cls_query"]
        TFM["small Transformer +<br/>alpha-head softmax over T"]
    end

    NEWLP["log pi_theta(u|s)<br/>fp32 logsoftmax"]
    RATIO["rho = exp(new − old)"]
    KLk3["k3 KL = (r−1) − log r<br/>r = pi_ref / pi_theta"]

    ALPHA["alpha_t (grad-tracking)"]
    PERTURN["r_hat_t = alpha_t · R"]
    TURNADV["A_turn (group-norm by t)"]
    TRAJADV["A_traj (group-norm by i)<br/>= (R−R_bar)/sigma_R  ← const wrt theta"]
    AH["A_H = alpha·A_traj + (1−alpha)·A_turn"]

    PPO["L_PPO = -mean min(rho·A_H,<br/>clip(rho,1±eps)·A_H)"]
    KLT["beta · KL_k3"]
    CONS["lambda · ||sum_t A_turn − A_traj||^2<br/>(tensor form, Method B only)"]
    TOTAL["L_total"]

    LORA -->|"forward"| NEWLP
    NEWLP --> RATIO
    OLD --> RATIO
    NEWLP --> KLk3
    REF --> KLk3

    CLS --> TFM
    TFM --> ALPHA
    ALPHA --> PERTURN
    R --> PERTURN
    PERTURN --> TURNADV
    R --> TRAJADV
    TURNADV --> AH
    TRAJADV --> AH
    TURNADV --> CONS
    TRAJADV --> CONS

    RATIO --> PPO
    AH --> PPO
    PPO --> TOTAL
    KLk3 --> KLT --> TOTAL
    CONS --> TOTAL

    TOTAL -. "dL/d theta_LoRA via NEWLP, KLk3, PPO" .-> LORA
    TOTAL -. "dL/d phi via alpha → r_hat → A_turn → (PPO + CONS)" .-> TFM
    TOTAL -. "dL/d phi" .-> CLS

    classDef const fill:#eee,stroke:#888,color:#333
    classDef trainable fill:#cfe,stroke:#2a7,color:#053
    classDef loss fill:#fcd,stroke:#c33,color:#600
    class OLD,REF,R const
    class LORA,CLS,TFM trainable
    class PPO,KLT,CONS,TOTAL loss
```

---

## 3. One `train_step(group)` — operation sequence

```mermaid
sequenceDiagram
    autonumber
    participant T as HGPOTrainer
    participant D as TurnRD
    participant P as LoRAPolicy
    participant Ref as PiRef
    participant O1 as AdamW_LoRA
    participant O2 as AdamW_TurnRD
    participant K as KLctrl

    T->>T: build_advantages(group)
    T->>D: decompose_with_grad(group)
    D-->>T: alpha KxT (grad), final_R K
    Note over T: per_turn_rewards = alpha * R<br/>compute A_traj, A_turn, A_H, L_cons

    T->>P: batched_logprobs(use_ref=False)
    P-->>T: new_logp (grad)

    T->>Ref: batched_logprobs(use_ref=True)
    Note over Ref: disable_adapter or swap SFT snapshot
    Ref-->>T: ref_logp (detached)

    Note over T: rho = exp(new - old)<br/>L_PPO = -min(rho*A, clip*A).mean()<br/>KL_k3 = (pi_ref/pi_theta - 1) - log(r)<br/>L = L_PPO + beta*KL + lambda*L_cons
    T->>T: (loss / grad_accum).backward()

    alt step boundary
        T->>P: clip_grad_norm LoRA params
        T->>O1: optimizer.step then zero_grad
        T->>D: clip_grad_norm TurnRD params
        T->>O2: optimizer.step then zero_grad
    else accumulate
        Note over T: keep grads accumulating
    end

    T->>K: update(observed_kl)
    K-->>T: beta_next
    Note over T,P: later weight_sync pushes new LoRA theta to vLLM
```

---

## Key things the diagrams encode

- **Two distinct forwards through the same Qwen body each step** — one with the LoRA active (grad on, gives `new_logp` → drives PPO + KL gradient into θ_LoRA), one with LoRA disabled (no_grad, gives `ref_logp` → constants for the KL term). Code: `_batched_logprobs` in `src/algorithms/grpo/trainer.py:312`.
- **Two separate optimizers** — `AdamW(θ_LoRA, lr=1e-6)` for the LLM and `AdamW(φ_TurnRD, lr=1e-4)` for the decomposer (`trainer.py:223-252`). They share `loss.backward()` but step independently.
- **Gradient enters TurnRD only via Method B** — the tensor-form `consistency_loss_tensor` plus the grad-tracking `α_t` flowing into `Â_turn` → `Â_H` → PPO loss. For Methods A/C, TurnRD doesn't exist and the decomposer is a pure-Python callable with zero grad.
- **`R_i` is a constant** w.r.t. both θ and φ (came from the env), but `α_t · R` makes the attribution of R across turns learnable.
- **PPO ratio uses `new − old`**, while **KL k3 uses `ref − new`** (reversed) — see `trainer.py:493-498`, ensuring an unbiased non-negative KL(π‖π_ref).
