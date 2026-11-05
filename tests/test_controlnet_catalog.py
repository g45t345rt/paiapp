"""Regression tests for the curated ControlNet catalog."""

import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import paths  # noqa: E402


class ControlNetCatalogTests(unittest.TestCase):
    def test_scanned_pool_entries_and_folder_fallback_are_listed_once(self):
        with tempfile.TemporaryDirectory() as root:
            folder = os.path.join(root, "controlnet")
            os.mkdir(folder)
            curated = os.path.join(root, "scanned.safetensors")
            fallback = os.path.join(folder, "fallback.pt")
            duplicate = os.path.join(folder, "duplicate.ckpt")
            for path in (curated, fallback, duplicate):
                open(path, "wb").close()
            settings = {
                "components": {
                    "controlnet": [
                        {"label": "Scanned", "path": curated},
                        {"label": "Duplicate", "path": duplicate},
                    ],
                },
            }
            with mock.patch.object(paths.settings, "section", return_value=settings), \
                    mock.patch.object(paths, "controlnet_dir", return_value=folder):
                self.assertEqual(paths.list_controlnets(), [curated, duplicate, fallback])


if __name__ == "__main__":
    unittest.main()
