"""Tests for kurtis.utils module"""

from unittest.mock import MagicMock, patch

import pytest

from kurtis.utils import clean_and_truncate, free_unused_memory, get_device


def test_get_device():
    """Test device detection logic"""
    # Test CPU fallback
    with patch("torch.cuda.is_available", return_value=False):
        with patch("torch.mps.is_available", return_value=False):
            assert get_device() == "cpu"

    # Test CUDA
    with patch("torch.cuda.is_available", return_value=True):
        assert get_device() == "cuda"

    # Test MPS
    with patch("torch.cuda.is_available", return_value=False):
        with patch("torch.mps.is_available", return_value=True):
            assert get_device() == "mps"


def test_clean_and_truncate_basic():
    """Test basic text cleaning and truncation"""
    tokenizer = MagicMock()
    # Mock tokenizer.encode(sentence, add_special_tokens=False) returning a list of IDs
    tokenizer.encode.return_value = [10, 20, 30]

    # Mock sent_tokenize to avoid dependency on NLTK data in this test
    with patch("kurtis.utils.sent_tokenize", return_value=["Sentence one.", "Sentence two."]):
        result = clean_and_truncate("Sentence one. Sentence two.", 100, tokenizer)

    assert result == "Sentence one. Sentence two."
    assert tokenizer.encode.called


def test_clean_and_truncate_truncation():
    """Test that text is truncated by sentences"""
    tokenizer = MagicMock()
    # First sentence fits, second doesn't
    tokenizer.encode.side_effect = [[1, 2, 3], [4, 5, 6, 7, 8]]

    with patch("kurtis.utils.sent_tokenize", return_value=["Short.", "This one is too long."]):
        # max_length = 5, first is 3 tokens, second is 5.
        result = clean_and_truncate("Short. This one is too long.", 5, tokenizer)

    assert result == "Short."


def test_clean_and_truncate_empty():
    """Test cleaning empty text"""
    tokenizer = MagicMock()
    assert clean_and_truncate("", 512, tokenizer) == ""
    # Note: text.strip() on None would fail, but let's see if we should handle it
    with pytest.raises(AttributeError):
        clean_and_truncate(None, 512, tokenizer)


def test_free_unused_memory():
    """Test memory freeing logic (doesn't crash)"""
    with patch("torch.cuda.empty_cache") as mock_cuda:
        with patch("gc.collect") as mock_gc:
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.backends.cuda.is_built", return_value=True):
                    free_unused_memory()
                    assert mock_gc.called
                    assert mock_cuda.called
