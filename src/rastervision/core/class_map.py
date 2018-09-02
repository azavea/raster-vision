class ClassItem(object):
    """A class id and associated data."""

    def __init__(self, id: int, name: str, color=None):
        """Construct a new ClassItem.

        Color is picked randomly if it is a null value.

        Args:
            id: (int) class id
            name: (string) name of the class
            color: (string) Pillow color code
        """
        self.id = id
        self.name = name
        self.color = color

    def __eq__(self, other):
        if isinstance(other, ClassItem):
            return (self.id == other.id and \
                    self.name == other.name and \
                    self.color == other.color)
        return False

    def __repr__(self):
        s = "CLASS ITEM: [{}] {}".format(self.id, self.name)
        if self.color:
            s += " ({})".format(self.color)
        return s


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

    def get_by_name(self, name):
        for item in self.get_items():
            if name == item.name:
                return item
        raise ValueError('{} is not a name in this ClassMap.'.format(name))

    def get_keys(self):
        """Return the keys."""
        return list(self.class_item_map.keys())

    def get_items(self):
        """Return list of ClassItems."""
        return list(self.class_item_map.values())

    def get_class_names(self):
        """Return list of class names sorted by id."""
        sorted_items = sorted(self.get_items(), key=lambda item: item.id)
        return [item.name for item in sorted_items]

    def __len__(self):
        return len(self.get_items())

    def has_all_colors(self):
        for item in self.get_items():
            if not item.color:
                return False
        return True

    def get_category_index(self):
        """Get the corresponding category_index used by TF Object Detection."""
        category_index = {}
        for class_item in self.get_items():
            category_index[class_item.id] = {
                'id': class_item.id,
                'name': class_item.name
            }
        return category_index
