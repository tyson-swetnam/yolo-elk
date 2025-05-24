"""
SAM2 Tracking Module

This module provides tracking capabilities for segmented elk objects
across video frames.
"""

from .track_manager import TrackManager
from .norfair_tracker import NorfairTracker

__all__ = ['TrackManager', 'NorfairTracker']
