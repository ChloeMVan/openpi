import time
import subprocess
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

@dataclass
class GPUMetrics:
    """GPU metrics snapshot."""
    timestamp: float
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    power_watts: Optional[float] = None
    temperature_c: Optional[float] = None

class SectionProfiler:
    """Profiler for individual sections with GPU monitoring."""
    
    def __init__(self):
        self.current_section = None
        self.section_start_time = None
        self.section_start_gpu = None
        self.gpu_monitor_thread = None
        self.gpu_samples = []
        self._stop_monitor = threading.Event()
    
    def _get_gpu_stats(self) -> Optional[GPUMetrics]:
        """Get current GPU stats using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout.strip():
                parts = result.stdout.strip().split(',')
                if len(parts) >= 3:
                    return GPUMetrics(
                        timestamp=time.time(),
                        utilization_percent=float(parts[0].strip()),
                        memory_used_mb=float(parts[1].strip()),
                        memory_total_mb=float(parts[2].strip()),
                        power_watts=float(parts[3].strip()) if len(parts) > 3 and parts[3].strip() else None,
                        temperature_c=float(parts[4].strip()) if len(parts) > 4 and parts[4].strip() else None
                    )
        except Exception:
            pass  # GPU monitoring might fail, that's OK
        return None
    
    def _monitor_gpu(self):
        """Background thread to monitor GPU during section execution."""
        self.gpu_samples = []
        while not self._stop_monitor.is_set():
            stats = self._get_gpu_stats()
            if stats:
                self.gpu_samples.append(stats)
            time.sleep(0.001)  # Sample every 1ms
    
    def start_section(self, name: str):
        """Start profiling a section."""
        print(f"PROFILER-SECTION: Starting '{name}'")
        self.current_section = name
        self.section_start_time = time.perf_counter()
        self.section_start_gpu = self._get_gpu_stats()
        
        # Start GPU monitoring in background
        self._stop_monitor.clear()
        self.gpu_monitor_thread = threading.Thread(target=self._monitor_gpu, daemon=True)
        self.gpu_monitor_thread.start()
    
    def end_section(self) -> Dict:
        """End profiling a section and return metrics."""
        print(f"PROFILER-SECTION: Ending '{self.current_section}'")

        if self.current_section is None:
            print("PROFILER-SECTION: No active section!")
            return {}
        
        # Stop GPU monitoring
        self._stop_monitor.set()
        if self.gpu_monitor_thread:
            self.gpu_monitor_thread.join(timeout=0.1)
        
        # Get end time and GPU stats
        end_time = time.perf_counter()
        end_gpu = self._get_gpu_stats()
        
        # Calculate metrics
        wall_time_ms = (end_time - self.section_start_time) * 1000
        
        # Calculate GPU utilization during section
        gpu_utilizations = [s.utilization_percent for s in self.gpu_samples if s]
        avg_gpu_util = sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else 0
        max_gpu_util = max(gpu_utilizations) if gpu_utilizations else 0
        
        # Calculate memory delta
        start_memory = self.section_start_gpu.memory_used_mb if self.section_start_gpu else 0
        end_memory = end_gpu.memory_used_mb if end_gpu else 0
        memory_delta_mb = end_memory - start_memory
        
        # Get peak memory during section
        memory_samples = [s.memory_used_mb for s in self.gpu_samples if s]
        peak_memory_mb = max(memory_samples) if memory_samples else 0
        
        result = {
            'section': self.current_section,
            'wall_time_ms': wall_time_ms,
            'gpu_utilization_avg_percent': avg_gpu_util,
            'gpu_utilization_max_percent': max_gpu_util,
            'gpu_memory_start_mb': start_memory,
            'gpu_memory_end_mb': end_memory,
            'gpu_memory_delta_mb': memory_delta_mb,
            'gpu_memory_peak_mb': peak_memory_mb,
        }
        print(f"PROFILER-SECTION: Result for '{self.current_section}': {result}")
        
        # Reset for next section
        self.current_section = None
        self.section_start_time = None
        self.section_start_gpu = None
        self.gpu_samples = []
        
        return result

class Pi0Profiler:
    """Main profiler for Pi0 model."""
    
    def __init__(self):
        self.section_profiler = SectionProfiler()
        self.current_timings = {}
    
    def time_section(self, name: str):
        """Context manager for timing a section."""
        class SectionTimer:
            def __init__(self, profiler, section_name):
                self.profiler = profiler
                self.name = section_name
            
            def __enter__(self):
                self.profiler.section_profiler.start_section(self.name)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                timing = self.profiler.section_profiler.end_section()
                if timing:
                    self.profiler.current_timings[self.name] = timing
                    print(f"PROFILER: Stored timing for '{self.name}': {timing}")
                else:
                    print(f"PROFILER: No timing returned for '{self.name}'") 
        
        return SectionTimer(self, name)
    
    def get_timings(self) -> Dict:
        """Get current timing data and reset."""
        print(f"PROFILER: get_timings() called. Current keys: {list(self.current_timings.keys())}")
        timings = self.current_timings
        print(f"PROFILER: Returning: {timings}")
        return self.current_timings.copy()

    def clear_timings(self):
        self.current_timings = {}
