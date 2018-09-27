from abc import (ABC, abstractmethod)


class Augmentor(ABC):
    """Defines a method for augmenting training data.
    """

    @abstractmethod
    def process(self, training_data, tmp_dir):
        """Augment training data.

        Args:
           training_data: The TrainingData to augment

        Returns:
           augmented TrainingData
        """
        pass
