"""Unit tests for MultiOmicsDataset."""

import pytest
import torch

from customics.datasets.multi_omics_dataset import MultiOmicsDataset


class TestMultiOmicsDataset:
    def test_len(self, omics_df, clinical_df, sample_ids):
        ds = MultiOmicsDataset(omics_df, clinical_df, sample_ids, "label", "OS", "OS.time")
        assert len(ds) == len(sample_ids)

    def test_getitem_shapes(self, omics_df, clinical_df, sample_ids):
        ds = MultiOmicsDataset(omics_df, clinical_df, sample_ids, "label", "OS", "OS.time")
        omics_tensors, lbl, os_time, os_event = ds[0]
        assert len(omics_tensors) == 2
        assert omics_tensors[0].shape == (50,)  # rna
        assert omics_tensors[1].shape == (30,)  # cnv
        assert isinstance(os_time, int)
        assert isinstance(os_event, int)

    def test_getitem_dtype(self, omics_df, clinical_df, sample_ids):
        ds = MultiOmicsDataset(omics_df, clinical_df, sample_ids, "label", "OS", "OS.time")
        omics_tensors, _, _, _ = ds[0]
        for t in omics_tensors:
            assert t.dtype == torch.float32

    def test_no_label(self, omics_df, clinical_df, sample_ids):
        ds = MultiOmicsDataset(omics_df, clinical_df, sample_ids, None, "OS", "OS.time")
        _, lbl, _, _ = ds[0]
        assert lbl == 0

    def test_get_samples(self, omics_df, clinical_df, sample_ids):
        ds = MultiOmicsDataset(omics_df, clinical_df, sample_ids, "label", "OS", "OS.time")
        assert ds.get_samples() == sample_ids
