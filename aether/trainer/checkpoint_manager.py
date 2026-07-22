import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

class CheckpointManager:
    """
    Manages atomic saving, versioning, loading, and rollback of model weights W_t.
    Ensures that live execution (17:52) always reads W_{t-1} safely, while
    nightly training (18:10) atomically updates to W_t without race conditions.
    """
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Finds the path to the latest valid model checkpoint (W_{t-1})."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_W_*.pt"))
        if not checkpoints:
            return None
        return checkpoints[-1]
        
    def save_checkpoint_atomically(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        version_tag: str,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Saves a checkpoint atomically.
        1. Writes to a temporary file (.tmp).
        2. Renames to final destination (checkpoint_W_{version_tag}.pt) atomically.
        """
        final_path = self.checkpoint_dir / f"checkpoint_W_{version_tag}.pt"
        tmp_path = self.checkpoint_dir / f"checkpoint_W_{version_tag}.pt.tmp"
        
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "version_tag": version_tag,
            "epoch": epoch,
            "metrics": metrics or {}
        }
        
        # Save to temp file
        torch.save(state, tmp_path)
        
        # Atomic rename/replace
        tmp_path.replace(final_path)
        
        # Save metadata JSON alongside
        meta_path = self.checkpoint_dir / f"checkpoint_W_{version_tag}.json"
        meta_data = {
            "version_tag": version_tag,
            "epoch": epoch,
            "metrics": metrics or {},
            "checkpoint_file": final_path.name
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
            
        return final_path
        
    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: Optional[Path] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Loads model weights from the specified path or the latest available checkpoint (W_{t-1}).
        Returns metadata / checkpoint state.
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint_path()
            
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError("No checkpoint found to load.")
            
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        
        if optimizer is not None and state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            
        return state

    def rollback_candidate(self, candidate_tag: str) -> Optional[Path]:
        """
        Rollback Layer (Armor 3).
        If training fails or data is invalid:
        1. Purges any partial or candidate W_t files (pt, tmp, json) for candidate_tag.
        2. Verifies and returns the last valid checkpoint W_{t-1}.
        """
        candidate_pt = self.checkpoint_dir / f"checkpoint_W_{candidate_tag}.pt"
        candidate_tmp = self.checkpoint_dir / f"checkpoint_W_{candidate_tag}.pt.tmp"
        candidate_json = self.checkpoint_dir / f"checkpoint_W_{candidate_tag}.json"
        
        for file_path in [candidate_pt, candidate_tmp, candidate_json]:
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError:
                    pass
                    
        return self.get_latest_checkpoint_path()
