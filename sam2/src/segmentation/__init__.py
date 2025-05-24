"""
SAM2 Segmentation Module

This module provides SAM2-based segmentation capabilities for elk detection
and analysis in video footage.
"""

from .sam2_segmenter import SAM2Segmenter
from .elk_segmenter import ElkSegmenter

__all__ = ['SAM2Segmenter', 'ElkSegmenter']
