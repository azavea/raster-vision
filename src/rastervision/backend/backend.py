from abc import ABC, abstractmethod


class Backend(ABC):
    """Functionality for a specific implementation of an MLTask.

    This should be subclassed to provide a bridge to third party ML libraries.
    There is a many-to-one relationship from backends to tasks.
    """

    @abstractmethod
    def process_scene_data(self, scene, data, tmp_dir):
        """Process each scene's training data

        Args:
            scene: Scene
            data: TrainingData

        Returns:
            backend-specific data-structures consumed by backend's
            process_sceneset_results
        """
        pass

    @abstractmethod
    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        """After all scenes have been processed, process the resultset

        Args:
            training_results: dependent on the ml_backend's process_scene_data
            validation_results: dependent on the ml_backend's
                process_scene_data
        """
        pass

    @abstractmethod
    def train(self, tmp_dir):
        """Train a model.
        """
        pass

    @abstractmethod
    def load_model(self, tmp_dir):
        """Load the model in preparation for one or more prediction calls."""
        pass

    @abstractmethod
    def predict(self, chips, windows, tmp_dir):
        """Return predictions for a chip using model.

        Args:
            chips: [[height, width, channels], ...] numpy array of chips
            windows: List of boxes that are the windows aligned with the chips.
        """
        pass
