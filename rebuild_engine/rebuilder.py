"""Model Rebuild Engine: ReasoningRebuildLoss (DPO with alpha=1.5 thinking multiplier and scheduled sampling)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object  # type: ignore
    Dataset = object  # type: ignore


if HAS_TORCH:
    class ReasoningRebuildLoss(nn.Module):
        """DPO Loss + Thinking-Token Amplification for My-Opus."""

        def __init__(self, beta: float = 0.1, thinking_weight: float = 1.5):
            super().__init__()
            self.beta = beta
            self.thinking_weight = thinking_weight

        def compute_weighted_logprobs(
            self, logits: torch.Tensor, labels: torch.Tensor, thinking_mask: torch.Tensor
        ) -> torch.Tensor:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = thinking_mask[:, 1:].contiguous()

            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            weights = torch.where(shift_mask == 1, self.thinking_weight, 1.0)
            return (token_log_probs * weights).sum(dim=-1)

        def forward(
            self,
            policy_chosen_logits: torch.Tensor,
            policy_rejected_logits: torch.Tensor,
            ref_chosen_logits: torch.Tensor,
            ref_rejected_logits: torch.Tensor,
            chosen_labels: torch.Tensor,
            rejected_labels: torch.Tensor,
            chosen_thinking_mask: torch.Tensor,
            rejected_thinking_mask: torch.Tensor,
        ) -> torch.Tensor:
            pi_chosen = self.compute_weighted_logprobs(
                policy_chosen_logits, chosen_labels, chosen_thinking_mask
            )
            pi_rejected = self.compute_weighted_logprobs(
                policy_rejected_logits, rejected_labels, rejected_thinking_mask
            )

            with torch.no_grad():
                ref_chosen = self.compute_weighted_logprobs(
                    ref_chosen_logits, chosen_labels, chosen_thinking_mask
                )
                ref_rejected = self.compute_weighted_logprobs(
                    ref_rejected_logits, rejected_labels, rejected_thinking_mask
                )

            pi_logratios = pi_chosen - pi_rejected
            ref_logratios = ref_chosen - ref_rejected
            return -F.logsigmoid(self.beta * (pi_logratios - ref_logratios)).mean()


    class ByteLevelCodeTokenizer:
        """Deterministic byte-level tokenizer with special control tokens for local training."""

        THINK_START = 256
        THINK_END = 257
        TOOL_START = 258
        TOOL_END = 259
        PAD_ID = 260
        VOCAB_SIZE = 32000

        def encode(self, text: str, max_len: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
            ids: List[int] = []
            mask: List[int] = []
            in_thinking = False

            i = 0
            while i < len(text) and len(ids) < max_len:
                if text.startswith("<thinking>", i):
                    ids.append(self.THINK_START)
                    mask.append(1)
                    in_thinking = True
                    i += len("<thinking>")
                elif text.startswith("</thinking>", i):
                    ids.append(self.THINK_END)
                    mask.append(1)
                    in_thinking = False
                    i += len("</thinking>")
                else:
                    ids.append(ord(text[i]) % 256)
                    mask.append(1 if in_thinking else 0)
                    i += 1

            while len(ids) < max_len:
                ids.append(self.PAD_ID)
                mask.append(0)

            return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


    class FlawTraceDataset(Dataset):
        """Loads DPO pairs from data/flaw_traces.jsonl."""

        def __init__(self, jsonl_path: Union[str, Path], max_len: int = 256):
            self.records: List[Dict[str, str]] = []
            self.tokenizer = ByteLevelCodeTokenizer()
            self.max_len = max_len
            path = Path(jsonl_path)
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        self.records.append(json.loads(line))

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            rec = self.records[idx]
            prompt = rec.get("prompt", "")
            chosen_text = f"{prompt}\n{rec.get('chosen_response', '')}"
            rejected_text = f"{prompt}\n{rec.get('rejected_response', '')}"

            c_ids, c_mask = self.tokenizer.encode(chosen_text, self.max_len)
            r_ids, r_mask = self.tokenizer.encode(rejected_text, self.max_len)
            return {
                "chosen_ids": c_ids,
                "chosen_mask": c_mask,
                "rejected_ids": r_ids,
                "rejected_mask": r_mask,
            }


    class ModelRebuilder:
        """Runs DPO + Scheduled Sampling to repair discovered reasoning flaws."""

        def __init__(self, model: nn.Module, lr: float = 1e-4, device: Optional[str] = None):
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model.to(self.device)
            self.ref_model = copy.deepcopy(model).to(self.device).eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

            self.criterion = ReasoningRebuildLoss(beta=0.1, thinking_weight=1.5)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        def apply_scheduled_sampling(
            self, input_ids: torch.Tensor, sample_prob: float = 0.15
        ) -> torch.Tensor:
            if sample_prob <= 0.0:
                return input_ids
            with torch.no_grad():
                logits = self.model(input_ids)
                pred_ids = torch.argmax(logits, dim=-1)
            noise_mask = (torch.rand_like(input_ids.float()) < sample_prob).long()
            return torch.where(noise_mask == 1, pred_ids, input_ids)

        def train_on_flaws(
            self,
            flaw_jsonl_path: Union[str, Path] = "data/flaw_traces.jsonl",
            epochs: int = 2,
            batch_size: int = 4,
            save_path: Optional[Union[str, Path]] = None,
        ) -> Dict[str, float]:
            dataset = FlawTraceDataset(flaw_jsonl_path)
            if len(dataset) == 0:
                return {"steps": 0, "initial_loss": 0.0, "final_loss": 0.0}

            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.model.train()

            initial_loss = None
            final_loss = 0.0
            steps = 0

            for epoch in range(epochs):
                sampling_rate = min(0.25, 0.05 * (epoch + 1))
                for batch in loader:
                    c_ids = batch["chosen_ids"].to(self.device)
                    c_mask = batch["chosen_mask"].to(self.device)
                    r_ids = batch["rejected_ids"].to(self.device)
                    r_mask = batch["rejected_mask"].to(self.device)

                    sampled_c_ids = self.apply_scheduled_sampling(c_ids, sampling_rate)

                    self.optimizer.zero_grad()
                    pi_c_logits = self.model(sampled_c_ids)
                    pi_r_logits = self.model(r_ids)

                    with torch.no_grad():
                        ref_c_logits = self.ref_model(c_ids)
                        ref_r_logits = self.ref_model(r_ids)

                    loss = self.criterion(
                        pi_c_logits, pi_r_logits,
                        ref_c_logits, ref_r_logits,
                        c_ids, r_ids,
                        c_mask, r_mask,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    loss_val = float(loss.item())
                    if initial_loss is None:
                        initial_loss = loss_val
                    final_loss = loss_val
                    steps += 1

            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), save_path)

            return {
                "steps": steps,
                "initial_loss": round(initial_loss or 0.0, 4),
                "final_loss": round(final_loss, 4),
            }
else:
    class ReasoningRebuildLoss:  # type: ignore
        pass

    class ByteLevelCodeTokenizer:  # type: ignore
        pass

    class FlawTraceDataset:  # type: ignore
        pass

    class ModelRebuilder:  # type: ignore
        pass
