"""
Object Tracking Module

This module contains classes and functions for multi-object tracking
using various tracking algorithms including Kalman filters and deep learning approaches.
"""

from .kalman_tracker import KalmanTracker
from .norfair_tracker import NorfairTracker
from .track_manager import TrackManager

__all__ = ['KalmanTracker', 'NorfairTracker', 'TrackManager']
