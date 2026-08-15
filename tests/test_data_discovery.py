"""
Unit & Integration Tests for Healthcare Claims Data Discovery Module
(Standard Library unittest)
"""

import os
import unittest
from src.data_discovery import HealthcareDataDiscovery


class TestHealthcareDataDiscovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.discovery = HealthcareDataDiscovery(base_dir=base_dir)
        cls.discovery.discover_files()
        cls.discovery.load_and_profile_files()
        cls.discovery.analyze_keys_and_target()
        cls.discovery.analyze_relationships_and_joins()

    def test_file_discovery(self):
        self.assertEqual(len(self.discovery.file_info), 8, "Expected exactly 8 CSV files discovered.")
        train_files = [f for f, info in self.discovery.file_info.items() if info["group"] == "TRAIN"]
        test_files = [f for f, info in self.discovery.file_info.items() if info["group"] == "TEST"]
        self.assertEqual(len(train_files), 4, "Expected 4 TRAIN files.")
        self.assertEqual(len(test_files), 4, "Expected 4 TEST files.")

    def test_load_and_profile(self):
        for filename, info in self.discovery.file_info.items():
            self.assertGreater(info["row_count"], 0, f"File {filename} should not be empty.")
            self.assertGreater(info["col_count"], 0, f"File {filename} should have columns.")
            self.assertIn("missing_stats", info, f"File {filename} missing stats not computed.")

    def test_keys_and_target_validation(self):
        self.assertEqual(self.discovery.provider_col, "Provider")
        self.assertEqual(self.discovery.bene_col, "BeneID")
        self.assertEqual(self.discovery.claim_col, "ClaimID")
        self.assertTrue(self.discovery.has_target_in_train, "TRAIN dataset must contain PotentialFraud column.")
        self.assertFalse(self.discovery.has_target_in_test, "TEST dataset must NOT contain PotentialFraud column.")

    def test_relationship_analysis(self):
        rel = self.discovery.rel_metrics
        self.assertEqual(rel["train"]["providers"], 5410)
        self.assertEqual(rel["test"]["providers"], 1353)
        self.assertEqual(rel["overlaps"]["provider_overlap_count"], 0, "Provider sets between train and test must be disjoint.")

    def test_report_generation(self):
        self.discovery.generate_reports()
        reports_dir = os.path.join(self.discovery.base_dir, "reports")
        self.assertTrue(os.path.exists(os.path.join(reports_dir, "dataset_assessment.md")))
        self.assertTrue(os.path.exists(os.path.join(reports_dir, "data_dictionary.csv")))
        self.assertTrue(os.path.exists(os.path.join(reports_dir, "dataset_relationships.md")))


if __name__ == "__main__":
    unittest.main()
