import json
from pathlib import Path
from typing import List, Optional

from loguru import logger

# Maps the ``world_short_name`` prefix in a task's ``archipelago.json`` to the eval
# "world type" bucket (ib / law / consulting). Used as the ``data_source`` so the
# per-dataset eval metrics report accuracy separately per category, e.g.
# ``eval/law/avg_score``, ``eval/ib/avg_score``, ``eval/consulting/avg_score``.
_WORLD_TYPE_BY_PREFIX = {
    "investment-banking": "ib",
    "law": "law",
    "management-consulting": "consulting",
}


def infer_world_type(task_path: Path) -> Optional[str]:
    """Infer the eval world type (ib/law/consulting) from a task's ``archipelago.json``.

    Reads the ``world_short_name`` field (e.g. ``law-world-433``,
    ``investment-banking-world-219``, ``management-consulting-world-129``) and maps
    its prefix to a category. Returns ``None`` when the file/field is missing or the
    prefix is unrecognized, so unknown tasks fall back to the default data source.
    """
    archipelago_file = task_path / "archipelago.json"
    if not archipelago_file.exists():
        return None
    try:
        with archipelago_file.open() as f:
            data = json.load(f)
            if isinstance(data, dict):
                world_short_name = data.get("world_short_name") or ""
            else:
                logger.warning(f"Invalid JSON structure in {archipelago_file}: expected a dictionary")
                return None
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read world_short_name from {archipelago_file}: {e}")
        return None
    for prefix, world_type in _WORLD_TYPE_BY_PREFIX.items():
        if world_short_name.startswith(prefix):
            return world_type
    logger.warning(f"Unrecognized world_short_name '{world_short_name}' in {archipelago_file}")
    return None


class HarborTaskDataset:
    """
    A dataset that loads Harbor task data from direct file/directory paths.
    Each dataset item is a path to a task directory.
    """

    def __init__(
        self,
        data_files: List[str],
    ):
        """
        Initialize the HarborTaskDataset.

        Args:
            data_files: List of direct file/directory paths pointing to Harbor task data
        """
        self.data_files = data_files

        # Load all data files
        self.task_paths = self._load_data_files()

        # Precompute the world type (ib/law/consulting) for each task so it can be
        # used as the data_source, giving per-category eval accuracy metrics.
        self.world_type_by_path = {str(task_path): infer_world_type(task_path) for task_path in self.task_paths}
        world_type_counts: dict = {}
        for world_type in self.world_type_by_path.values():
            world_type_counts[world_type] = world_type_counts.get(world_type, 0) + 1

        logger.info(
            f"HarborTaskDataset initialized with {len(self.task_paths)} task paths; "
            f"world type breakdown: {world_type_counts}"
        )

    def _data_source_for(self, task_path: Path) -> str:
        """Data source for a task: its world type, falling back to the task path."""
        return self.world_type_by_path.get(str(task_path)) or str(task_path)

    def _load_data_files(self) -> List[Path]:
        """Load all data files from direct paths and return list of task paths."""
        task_paths = []

        for data_source in self.data_files:
            source_path = Path(data_source)

            if not source_path.exists():
                logger.warning(f"Path does not exist: {data_source}")
                continue

            logger.info(f"Loading data from: {data_source}")

            # If the path is a directory, find all valid task subdirectories
            if source_path.is_dir():
                # Look for task subdirectories and validate them
                all_dirs = [d for d in source_path.iterdir() if d.is_dir()]
                valid_task_dirs = [d for d in all_dirs if self._is_valid_task_directory(d)]

                if valid_task_dirs:
                    task_paths.extend(valid_task_dirs)
                    logger.info(
                        f"Found {len(valid_task_dirs)} valid task directories out of {len(all_dirs)} total directories"
                    )
                elif self._is_valid_task_directory(source_path):
                    # If no subdirectories but the main directory is valid, treat it as a task
                    task_paths.append(source_path)
                    logger.info("Using main directory as valid task")
                else:
                    logger.warning(f"No valid task directories found in {source_path}")
            else:
                # If it's a file, treat it as a single task (files can't be valid task directories)
                logger.warning(f"File {source_path} cannot be a valid task directory (missing instruction.md)")

        return task_paths

    def _is_valid_task_directory(self, task_path: Path) -> bool:
        """Check if a directory is a valid task directory (has instruction.md file)."""
        if not task_path.is_dir():
            return False

        instruction_file = task_path / "instruction.md"
        return instruction_file.exists() and instruction_file.is_file()

    def __getitem__(self, index: int) -> dict:
        """Get a task path by index as a dictionary with 'prompt', 'env_class', and 'env_extras' keys."""
        if index >= len(self.task_paths):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.task_paths)}")
        task_path = self.task_paths[index]
        return {
            "prompt": str(task_path),
            "env_class": None,
            "env_extras": {"data_source": self._data_source_for(task_path), "task_path": str(task_path)},
            "uid": str(index),
        }

    def __len__(self) -> int:
        """Return the number of tasks in the dataset."""
        return len(self.task_paths)

    def __iter__(self):
        """Iterate over all task paths as dictionaries."""
        for index, task_path in enumerate(self.task_paths):
            yield {
                "prompt": str(task_path),
                "env_class": None,
                "env_extras": {"data_source": self._data_source_for(task_path), "task_path": str(task_path)},
                "uid": str(index),
            }

    def get_task_paths(self) -> List[Path]:
        """Return all task paths as a list."""
        return self.task_paths.copy()

    def collate_fn(self, item_list):
        """Collate function for batching task dictionaries."""
        return item_list
