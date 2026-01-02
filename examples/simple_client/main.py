import dataclasses
import enum
import logging
import pathlib
import time
import json  # Add this import

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import polars as pl
import rich
import tqdm
import tyro

logger = logging.getLogger(__name__)


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    host: str = "0.0.0.0"
    # Port to connect to the server. If None, the server will use the default port.
    port: int | None = 8000
    # API key to use for the server.
    api_key: str | None = None
    # Number of inference iterations to run.
    num_steps: int = 20
    # Environment to run the policy in.
    env: EnvMode = EnvMode.ALOHA_SIM


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)),
            "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)),
            "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)),
            "p99": float(np.quantile(times, 0.99)),
        }

    def print_all_stats(self) -> None:
        """Print statistics for all keys in a concise format."""

        table = rich.table.Table(
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )

        # Add metric column with custom styling
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)

        # Add statistical columns with consistent styling
        stat_columns = [
            ("Mean", "yellow", "mean"),
            ("Std", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]

        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)

        # Add rows for each metric with formatted values
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        # Print with custom console settings
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def main(args: Args) -> None:
    obs_fn = {
        EnvMode.ALOHA: _random_observation_aloha,
        EnvMode.ALOHA_SIM: _random_observation_aloha,
        EnvMode.DROID: _random_observation_droid,
        EnvMode.LIBERO: _random_observation_libero,
    }[args.env]

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    # Send a few observations to make sure the model is loaded.
    logger.info("Warming up model...")
    for _ in range(2):
        policy.infer(obs_fn())

    timing_recorder = TimingRecorder()
    logger.info(f"Running {args.num_steps} inference iterations...")

    for i in tqdm.trange(args.num_steps, desc="Running policy"):
        inference_start = time.time()
        action = policy.infer(obs_fn())
        
        # Record client inference time
        timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
        
        # Record server timings
        for key, value in action.get("server_timing", {}).items():
            timing_recorder.record(f"server_{key}", value)
        
        # Record component timings (if available)
        component_timings = action.get("component_timings", {})
        if component_timings:
            for component, metrics in component_timings.items():
                # Wall time
                wall_time = metrics.get('wall_time_ms', 0)
                if wall_time > 0:
                    timing_recorder.record(f"component_{component}_wall_ms", wall_time)
                
                # GPU utilization
                gpu_avg = metrics.get('gpu_utilization_avg_percent', 0)
                gpu_max = metrics.get('gpu_utilization_max_percent', 0)
                if gpu_avg > 0 or gpu_max > 0:
                    timing_recorder.record(f"component_{component}_gpu_avg_%", gpu_avg)
                    timing_recorder.record(f"component_{component}_gpu_max_%", gpu_max)
                
                # GPU memory
                mem_delta = metrics.get('gpu_memory_delta_mb', 0)
                mem_peak = metrics.get('gpu_memory_peak_mb', 0)
                if mem_delta != 0 or mem_peak > 0:
                    timing_recorder.record(f"component_{component}_mem_delta_mb", mem_delta)
                    timing_recorder.record(f"component_{component}_mem_peak_mb", mem_peak)
            
            # Batch size
            batch_size = action.get("batch_size", 1)
            timing_recorder.record("batch_size", batch_size)

    # Print statistics
    timing_recorder.print_all_stats()
    
    # Always save results
    output_dir = pathlib.Path("./benchmark_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parquet_path = output_dir / f"pi0_timing_{timestamp}.parquet"
    json_path = output_dir / f"pi0_summary_{timestamp}.json"
    
    timing_recorder.write_parquet(parquet_path)
    
    # Save JSON summary
    summary = {}
    for key in timing_recorder._timings.keys():
        if timing_recorder._timings[key]:
            stats = timing_recorder.get_stats(key)
            summary[key] = stats
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("RESULTS SAVED:")
    print(f"  Parquet: {parquet_path}")
    print(f"  JSON:    {json_path}")
    print("="*60)
    
    # Quick component summary
    component_cols = [k for k in timing_recorder._timings.keys() if k.startswith("component_")]
    if component_cols:
        print("\nCOMPONENT TIMING COLLECTED:")
        for col in sorted(component_cols):
            if "wall_ms" in col:
                times = timing_recorder._timings[col]
                if times:
                    component_name = col.replace("component_", "").replace("_wall_ms", "")
                    mean = np.mean(times)
                    p99 = np.percentile(times, 99)
                    print(f"  {component_name}: {mean:.1f}ms mean, {p99:.1f}ms P99")

def _random_observation_aloha() -> dict:
    """Create random observation for ALOHA environment."""
    # Note: Batch size is 1 here (single observation)
    return {
        "state": np.ones((14,)),  # Shape: (14,) = batch size 1
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _random_observation_droid() -> dict:
    """Create random observation for DROID environment."""
    # Batch size 1
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),  # Shape: (7,) = batch 1
        "observation/gripper_position": np.random.rand(1),  # Shape: (1,) = batch 1
        "prompt": "do something",
    }


def _random_observation_libero() -> dict:
    """Create random observation for LIBERO environment."""
    # Batch size 1
    return {
        "observation/state": np.random.rand(8),  # Shape: (8,) = batch 1
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))