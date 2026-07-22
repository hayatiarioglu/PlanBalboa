import unittest
import tempfile
import shutil
import torch
import torch.nn as nn
from pathlib import Path
from aether.trainer.checkpoint_manager import CheckpointManager

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2)

class TestCheckpointVersioning(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.manager = CheckpointManager(self.test_dir)
        self.model = DummyModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_atomic_checkpoint_save_and_load(self):
        """
        Tests atomic saving of W_t and loading into model.
        """
        # Change weight to a known value
        with torch.no_grad():
            self.model.fc.weight.fill_(4.2)

        saved_path = self.manager.save_checkpoint_atomically(
            model=self.model,
            optimizer=self.optimizer,
            version_tag="2026_W29",
            epoch=10,
            metrics={"loss": 0.123}
        )

        self.assertTrue(saved_path.exists())
        self.assertFalse((Path(self.test_dir) / "checkpoint_W_2026_W29.pt.tmp").exists())

        # Reset model weights
        new_model = DummyModel()
        with torch.no_grad():
            new_model.fc.weight.fill_(0.0)

        # Load saved checkpoint
        state = self.manager.load_checkpoint(new_model, saved_path)

        self.assertEqual(state["version_tag"], "2026_W29")
        self.assertEqual(state["epoch"], 10)
        self.assertAlmostEqual(state["metrics"]["loss"], 0.123)
        self.assertTrue(torch.allclose(new_model.fc.weight, torch.full_like(new_model.fc.weight, 4.2)))

    def test_get_latest_checkpoint(self):
        """
        Tests that get_latest_checkpoint_path returns W_{t-1} sequentially.
        """
        self.assertIsNone(self.manager.get_latest_checkpoint_path())

        self.manager.save_checkpoint_atomically(self.model, self.optimizer, version_tag="2026_W01")
        self.manager.save_checkpoint_atomically(self.model, self.optimizer, version_tag="2026_W02")
        self.manager.save_checkpoint_atomically(self.model, self.optimizer, version_tag="2026_W03")

        latest_path = self.manager.get_latest_checkpoint_path()
        self.assertIsNotNone(latest_path)
        self.assertTrue(latest_path.name.endswith("checkpoint_W_2026_W03.pt"))

    def test_checkpoint_rollback(self):
        """
        Tests Armor 3: Rollback mechanism on training failure.
        """
        # Save valid W_{t-1}
        prev_path = self.manager.save_checkpoint_atomically(self.model, self.optimizer, version_tag="2026_W01")
        
        # Simulate partial failed save for W_t (e.g. creating corrupt/tmp files)
        failed_tag = "2026_W02"
        tmp_file = Path(self.test_dir) / f"checkpoint_W_{failed_tag}.pt.tmp"
        corrupt_pt = Path(self.test_dir) / f"checkpoint_W_{failed_tag}.pt"
        tmp_file.write_text("CORRUPT_TMP_DATA")
        corrupt_pt.write_text("CORRUPT_PT_DATA")
        
        self.assertTrue(tmp_file.exists())
        self.assertTrue(corrupt_pt.exists())
        
        # Trigger rollback
        rolled_back_path = self.manager.rollback_candidate(failed_tag)
        
        # Verify candidate files were purged
        self.assertFalse(tmp_file.exists())
        self.assertFalse(corrupt_pt.exists())
        
        # Verify system rolled back to untouched W_{t-1}
        self.assertEqual(rolled_back_path, prev_path)

if __name__ == "__main__":
    unittest.main()
