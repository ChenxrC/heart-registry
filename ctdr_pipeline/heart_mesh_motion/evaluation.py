"""Frozen blocked-time holdout and support-only interpolation."""
from __future__ import annotations
import numpy as np


def make_split(frame_count: int,run_id: str) -> dict:
    if type(frame_count) is not int or frame_count<6:raise ValueError('At least six integer-indexed frames required')
    if run_id=='series_01' and frame_count!=31:raise ValueError('Series 1 expected 31 frames')
    if run_id=='series_02' and frame_count!=32:raise ValueError('Series 2 expected 32 frames')
    start,end=frame_count//3,2*frame_count//3
    holdout=list(range(start,end))
    return {'split_id':'blocked_middle_third','support_indices':[i for i in range(frame_count) if i not in holdout],
        'holdout_indices':holdout,'frame_count':frame_count,'run_id':run_id}


def interpolate_parameters(support_times,parameters,query_times) -> np.ndarray:
    time=np.asarray(support_times,dtype=float);values=np.asarray(parameters,dtype=float);query=np.asarray(query_times,dtype=float)
    if time.ndim!=1 or len(time)<2 or not np.isfinite(time).all() or np.any(np.diff(time)<=0):raise ValueError('Strictly increasing support time required')
    if values.ndim!=2 or len(values)!=len(time) or not np.isfinite(values).all() or query.ndim!=1 or not np.isfinite(query).all():raise ValueError('Invalid interpolation input')
    if np.any(query<time[0]) or np.any(query>time[-1]):raise ValueError('No extrapolation outside observed support range')
    return np.column_stack([np.interp(query,time,values[:,i]) for i in range(values.shape[1])]) if values.shape[1] else np.empty((len(query),0))
