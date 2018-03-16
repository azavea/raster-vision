class LabelItem(object):
    """A label id and associated data."""

    def __init__(self, id, name):
        """Construct a new LabelItem.

        Args:
            id: (int) id of a label
            name: (string) name of label
        """
        self.id = id
        self.name = name


class LabelMap(object):
    """A map from label_id to LabelItem."""

    def __init__(self, label_items):
        """Construct a new LabelMap.

        Args:
            label_items: list of LabelItems
        """
        self.label_item_map = {}
        for label_item in label_items:
            self.label_item_map[label_item.id] = label_item

    def get_by_id(self, id):
        """Return a LabelItem by its id.

        Args:
            id: (int) id of label
        """
        return self.label_item_map[id]

    def get_items(self):
        """Return list of LabelItems."""
        return self.label_item_map.values()

    def __len__(self):
        return len(self.get_items())

    def get_category_index(self):
        """Get the corresponding category_index used by TF Object Detection."""
        category_index = {}
        for label_item in self.get_items():
            category_index[label_item.id] = {
                'id': label_item.id,
                'name': label_item.name
            }
        return category_index
