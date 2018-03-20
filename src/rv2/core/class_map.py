class ClassItem(object):
    """A class id and associated data."""

    def __init__(self, id, name):
        """Construct a new ClassItem.

        Args:
            id: (int) class id
            name: (string) name of the class
        """
        self.id = id
        self.name = name


class ClassMap(object):
    """A map from class_id to ClassItem."""

    def __init__(self, class_items):
        """Construct a new ClassMap.

        Args:
            class_items: list of ClassItems
        """
        self.class_item_map = {}
        for class_item in class_items:
            self.class_item_map[class_item.id] = class_item

    def get_by_id(self, id):
        """Return a ClassItem by its id.

        Args:
            id: (int) id of class
        """
        return self.class_item_map[id]

    def get_items(self):
        """Return list of ClassItems."""
        return self.class_item_map.values()

    def __len__(self):
        return len(self.get_items())

    def get_category_index(self):
        """Get the corresponding category_index used by TF Object Detection."""
        category_index = {}
        for class_item in self.get_items():
            category_index[class_item.id] = {
                'id': class_item.id,
                'name': class_item.name
            }
        return category_index
